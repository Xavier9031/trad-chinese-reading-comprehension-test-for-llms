# trad-chinese-reading-comprehension-test-for-llms-NYCU312707024

## 1. Introduction
本次作業的主題為「Traditional Chinese Reading Comprehension Test for LLMs」，旨在探討大型語言模型在傳統中文閱讀理解能力上的提升潛力。我們將對大型語言模型進行訓練，使其能夠準確回答基於傳統中文文章的問題。我們選用Chinese-LLaMA-Alpaca-2提供的Chinese-LLaMA-2-7B的模型進行微調，並將其應用於閱讀理解任務。我們將使用的數據集包含了大量的中文文章、問題和答案，並且已經經過了人工標註。我們將使用此數據集來訓練模型，使其能夠準確回答基於傳統中文文章的問題。我們的目標是提升模型在閱讀理解方面的能力，特別是針對選擇題的解答。
## 2. Environment
### 2.1 環境需求
參考助教提供的環境設定，使用以下版本的套件：

```bash
git clone https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
```
（重啟終端機）
```bash
conda create --name llm python=3.10
conda activate llm
cd Chinese-LLaMA-Alpaca-2
pip install -r requirement.txt
```
### 2.2 檔案修改

將 scripts/training/run_clm_sft_with_peft.py
line 340 行處 tokenizer 檢查部分進行註解
```python
# if (len(tokenizer)) != 55296:
#     raise ValueError(f"The vocab size of the tokenizer should be 55296, but found {len(tokenizer)}.\n"
#                      "Please use Chinese-LLaMA-2 tokenizer.")
```

## 3. Dataset

### 3.1 Data Format

本次作業的數據集含以下幾個欄位：

1. **ID**：每筆數據的唯一識別碼。
2. **文章**：提供背景或情境的文本，與後續問題相關。
3. **問題**：基於文章內容的問題。
4. **選項1** - **選項4**：針對問題的四個選擇答案。
5. **正確答案**：表示哪一個選項（1, 2, 3, 或 4）是正確的答案。
6. **資料來源**：提供文章內容的來源資訊。

此數據集可以用於訓練機器學習模型進行閱讀理解和問題回答的任務，數據的結構使其適合於自然語言處理應用，特別是在理解上下文和提取信息方面。
### 3.2 Data Preprocessing (xlsx to json)

這段程式碼的功能是從 Excel 文件中讀取數據，並將其轉換成特定格式的 JSON 文件。使用者可以通過命令行參數指定輸入文件的路徑、輸出文件的路徑、一段指示文字，以及選擇使用全部數據還是只使用數據的一部分（通過設置抽樣比率）。

### 使用說明

1. **設定輸入文件路徑** (`--file_path`): 指定要讀取的 Excel 文件路徑。預設為 `AI.xlsx`。
2. **設定輸出文件路徑** (`--output_path`): 指定輸出的 JSON 文件路徑。預設為 `AI.json`。
3. **設定指示文字** (`--instruction`): 設定在 JSON 文件中每個條目中顯示的指示文字。預設文字為 `請根據以下輸入內容詳細思考後，回答選擇題，並只以阿拉伯數字回答：\n`。
4. **設定抽樣比率** (`--sample_rate`): 決定從 Excel 文件中抽取多少比例的數據來生成 JSON 文件。若設為 1，則使用全部數據；若設為 0.2，則隨機選取 20% 的數據。

### 命令行示例

- **處理全部數據**:
  ```
  python process_json.py --file_path="AI.xlsx" --output_path="AI.json"
  ```

- **處理 20% 的測試數據**:
  ```
  python process_json.py --file_path="AI.xlsx" --output_path="AI_test.json" --sample_rate=0.2
  ```

## 4. Model

## 模型介紹

由於設備限制，本次作業使用的模型是基於 **Chinese-LLaMA-2-7B 基座模型** 進行的微調（Supervised Fine-Tuning，簡稱SFT）。該模型是由中文LLaMA&Alpaca大模型的第二期項目開發的，具體特點如下：

- **語言專注**：該模型專門針對中文語言進行優化，具備強大的中文處理能力。

- **基座模型**：這是一種預訓練模型，提供了豐富的語言理解基礎，可以通過微調適應特定的下游任務。

- **FlashAttention-2技術**：採用了高效的注意力機制，使模型在處理長文本時更為高效，同時優化了显存占用。

- **長上下文支持**：此模型能夠處理長達4K的文本上下文，適合於閱讀理解和其他需要長文本處理的任務。

下載模型及更多資訊，請參考其[官方GitHub頁面](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)。

### 微調（SFT）過程

進行SFT是為了使基座模型更好地適應特定的應用場景。在本作業中，微調的目的是提升模型在語義理解方面的能力，特別是針對選擇題的解答。通過使用特定的訓練數據，模型能夠學習到如何更準確地理解問題並選擇正確的答案。

### 微調（SFT）詳細操作
1. 完整下載中文LLaMA&Alpaca大模型的第二期項目，並將其放置於指定目錄下

2. 修改run_sft.sh參數
```bash
lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

root=/mnt/htchang/ML/HW4
pretrained_model=${root}/LLAMA_model/chinese-llama-2-7b-hf-20231207T072133Z-002/chinese-llama-2-7b-hf
chinese_tokenizer_path=${root}/LLAMA_model/chinese-llama-2-lora-7b/tokenizer.model
dataset_dir=${root}/trad-chinese-reading-comprehension-test-for-llms
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
max_seq_length=512
output_dir=my_first_lora_model
validation_file=${root}/trad-chinese-reading-comprehension-test-for-llms/AI_test.json

deepspeed_config_file=ds_zero2_no_offload.json
```
3. 執行run_sft.sh

4. 將輸出的模型放置於指定目錄下

5. 合併模型
在執行生成腳本前，必須要先將原模型(base_model)與訓練出來的LoRA模型(lora_model)合併
[參考官方文件](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/manual_conversion_zh)
    * 找到輸出的`sft_lora_model`
    * 找到裡面的`adapter_config.json`
    * 刪掉裡面的`"enable_lora": null`, `"merge_weights": false`
    * 執行下方程式進行合併
    
    ```json
    python scripts/merge_llama2_with_chinese_lora_low_mem.py \
        --base_model  /mnt/htchang/HW4/chinese-llama-2-7b \
        --lora_model /mnt/htchang/HW4/Chinese-LLaMA-Alpaca-2/scripts/training/my_bonus_lora_model_7b/sft_lora_model \
        --output_type huggingface \
        --output_dir ./tt_7b_bonus
    ```

## Inference (generate_ans.py)

### 批次生成腳本使用說明

此腳本用於從JSON文件中讀取數據，並使用預訓練的LLaMA模型進行批次文本生成，最後將生成的答案保存到CSV文件中。您可以自行修改此腳本，以整理成符合您需求的數據格式。

### 程式功能

- 從指定的JSON文件中讀取數據。
- 使用LLaMA模型根據讀取的數據生成文本回應。
- 將生成的回應保存到CSV文件中。

### 使用步驟

1. **準備環境**：
   確保您的Python環境已安裝以下依賴：
   - `torch`
   - `transformers`

2. **指定模型和設定**：
   使用命令行參數來指定基座模型的路徑和其他設定，例如：
   ```bash
   python generate_ans.py --base_model='path_to_base_model' --gpus='0' --load_in_8bit
   ```

3. **執行腳本**：
   執行腳本後，程式將讀取指定的JSON文件，並使用LLaMA模型生成文本回應。

4. **查看結果**：
   生成的回應將被保存在CSV文件中。您可以打開該文件查看結果。

### 參數說明

- `--base_model`：指定LLaMA模型的路徑。
- `--gpus`：指定使用的GPU。預設為"0"，表示使用`cuda:0`。若要使用多個GPU，則格式為`--gpus=0,1,...`。
- `--load_in_8bit`：若設定此選項，則使用8位量化模型。
- `--load_in_4bit`：若設定此選項，則使用4位量化模型。
### 小巧思：
有人號稱假裝給GPT小費，能讓它產生品質更高，錢越多效果越好！
[靈感來源](https://twitter.com/voooooogel/status/1730726744314069190)
```python
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant, and you will receive an additional bonus of $20 for each correct answer. 你是一個樂於助人的助手，每答對一題都會多獲得20美元的獎金。\n"
```

## 5. Evaluation
* Private Leaderboard: 0.81
* Public Leaderboard: 0.82666

## 6. 心得

這次「Traditional Chinese Reading Comprehension Test for LLMs」的作業，真是一次特別的經歷。說實話，一開始我對於處理這麼複雜的技術任務感到有點頭大，但進行過程中卻意外地發現自己對挑戰的興趣。

在這次的實作過程中，我學習到了如何運用大型語言模型（LLMs）來解決傳統中文閱讀理解問題。整個過程涉及了從選擇適合的模型、理解數據集結構、進行數據預處理、微調模型到評估模型性能等多個環節。

最讓我印象深刻的是那些摸索和實驗的時刻。每當碰到數據格式不對或模型沒有預期的表現時，我就得想辦法解決問題。這過程中，我學到了不少新招數，也讓我對自己的解決問題能力更有信心。

跟模型打交道的過程也挺有趣的。雖然有時候會讓人有點沮喪（比如當模型不按我想的那樣工作時），但當一切都順利運行時，那種成就感真的很棒。而且，能看到自己調整的模型逐漸進步，這感覺真的很酷。

總之，這次作業不僅提高了我的技術能力，更讓我對大型語言模型在處理中文閱讀理解問題上的應用有了實踐的機會。這次的學習經驗無疑將對我未來在自然語言處理領域的探索和發展大有裨益。

