# 以下範例為輸入json用以批次生成的腳本，並將結果存於 ./answer.csv
# 你也可以自行修改腳本，整理成自己習慣的資料格式(txt..)用於批次輸入
import torch
import os
import argparse
import json,csv
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    BitsAndBytesConfig
)

# 訓練時System Prompt可以自訂，生成時建議與訓練的Prompt一致
# 請參考script/training/build_dataset.py 進行Prompt的調整
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant, and you will receive an additional bonus of $20 for each correct answer. 你是一個樂於助人的助手，每答對一題都會多獲得20美元的獎金。\n"

TEMPLATE_WITH_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n{instruction} [/INST]"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--base_model',
    default=None,
    type=str,
    required=True, 
    help='Base model path')
parser.add_argument(
    '--gpus',
    default="0",
    type=str,
    help='If None, cuda:0 will be used. Inference using multi-cards: --gpus=0,1,... ')
parser.add_argument(
    '--load_in_8bit',
    action='store_true',
    help='Use 8 bit quantized model')
parser.add_argument(
    '--load_in_4bit',
    action='store_true',
    help='Use 4 bit quantized model')
args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# Get GPU devices
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Model
model = LlamaForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=torch.float16
    )
)

# Load Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

# Do inference
with open('trad-chinese-reading-comprehension-test-for-llms/AI1000.json', 'r') as file:
    json_data = json.load(file)
    with open('answer_bonus_7b_finetune.csv', 'w', newline='', encoding='utf-8') as csv_file:
        writer=csv.writer(csv_file)
        writer.writerow(['ID', 'Answer'])
        for row in json_data:
            id = row['id']
            instruction = row['instruction']+ '\n' + row['input'] +"\n",

            prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map({'instruction': instruction,'system_prompt': DEFAULT_SYSTEM_PROMPT})

            # print("prompt:\n"+prompt)
            inputs = tokenizer.encode(prompt+'\n', return_tensors="pt").to(DEV)

            generate_kwargs = dict(
                input_ids=inputs,
                temperature=0.1,
                top_p=0.9,
                top_k=15,
                do_sample=True,
                max_new_tokens=1, #為了回答選擇題而設定1
                repetition_penalty=1.1,
                guidance_scale=1.0
            )
            outputs = model.generate(**generate_kwargs)
            # print("outputs:\n"+str(outputs))
            result = tokenizer.decode(outputs[0])
            # print("result:\n"+result)
            response = result.split('[/INST]\n')[-1]
            print("response:\n"+response)
            writer.writerow([id, response[0]])