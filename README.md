# trad-chinese-reading-comprehension-test-for-llms-NYCU312707024

## 1. Introduction
本次作業的主題為「Traditional Chinese Reading Comprehension Test for LLMs」，旨在探討大型語言模型在傳統中文閱讀理解能力上的提升潛力。我們將對多個大型語言模型進行訓練，使其能夠準確回答基於傳統中文文章的問題。本次研究不僅包括先前選用的Chinese-LLaMA-Alpaca-2模型，還**新增了最新的LLaMA-3-8B及Chinese-LLaMA-3-8B模型進行微調**，並將其應用於閱讀理解任務。我們將使用包含大量中文文章、問題和答案的數據集，這些數據集均經過人工標註，並且覆蓋廣泛的主題和文本類型。訓練模型的目的是使其能夠準確回答基於傳統中文文章的問題，提升模型在閱讀理解方面的能力，特別是針對選擇題的解答。通過引入更先進的模型和更豐富的數據集，我們期望能進一步提升大型語言模型在傳統中文閱讀理解任務中的表現，為未來的自然語言處理研究提供重要的數據和方法論支持。

**下面將會分成兩個部分來介紹我們的作業，分別是Chinese-LLaMA-Alpaca-2、Chinese-LLaMA-Alpaca-3。**
## 2. Chinese-LLaMA-Alpaca-2
### 2.1 Environment
#### 2.1.1 環境需求
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
#### 2.1.2 檔案修改

將 scripts/training/run_clm_sft_with_peft.py
line 340 行處 tokenizer 檢查部分進行註解
```python
# if (len(tokenizer)) != 55296:
#     raise ValueError(f"The vocab size of the tokenizer should be 55296, but found {len(tokenizer)}.\n"
#                      "Please use Chinese-LLaMA-2 tokenizer.")
```

## 2.2 Dataset

### 2.2.1 Data Format

本次作業的數據集含以下幾個欄位：

1. **ID**：每筆數據的唯一識別碼。
2. **文章**：提供背景或情境的文本，與後續問題相關。
3. **問題**：基於文章內容的問題。
4. **選項1** - **選項4**：針對問題的四個選擇答案。
5. **正確答案**：表示哪一個選項（1, 2, 3, 或 4）是正確的答案。
6. **資料來源**：提供文章內容的來源資訊。

此數據集可以用於訓練機器學習模型進行閱讀理解和問題回答的任務，數據的結構使其適合於自然語言處理應用，特別是在理解上下文和提取信息方面。
### 2.2.2 Data Preprocessing (xlsx to json)

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

### 2.3. Model

#### 2.3.1模型介紹

由於設備限制，本次作業使用的模型是基於 **Chinese-LLaMA-2-7B 基座模型** 進行的微調（Supervised Fine-Tuning，簡稱SFT）。該模型是由中文LLaMA&Alpaca大模型的第二期項目開發的，具體特點如下：

- **語言專注**：該模型專門針對中文語言進行優化，具備強大的中文處理能力。

- **基座模型**：這是一種預訓練模型，提供了豐富的語言理解基礎，可以通過微調適應特定的下游任務。

- **FlashAttention-2技術**：採用了高效的注意力機制，使模型在處理長文本時更為高效，同時優化了显存占用。

- **長上下文支持**：此模型能夠處理長達4K的文本上下文，適合於閱讀理解和其他需要長文本處理的任務。

下載模型及更多資訊，請參考其[官方GitHub頁面](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)。

#### 2.3.2 微調（SFT）過程

進行SFT是為了使基座模型更好地適應特定的應用場景。在本作業中，微調的目的是提升模型在語義理解方面的能力，特別是針對選擇題的解答。通過使用特定的訓練數據，模型能夠學習到如何更準確地理解問題並選擇正確的答案。

#### 2.3.3 微調（SFT）詳細操作
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

### 2.4 Inference (generate_ans.py)

#### 2.4.1批次生成腳本使用說明

此腳本用於從JSON文件中讀取數據，並使用預訓練的LLaMA模型進行批次文本生成，最後將生成的答案保存到CSV文件中。您可以自行修改此腳本，以整理成符合您需求的數據格式。

#### 2.4.2程式功能

- 從指定的JSON文件中讀取數據。
- 使用LLaMA模型根據讀取的數據生成文本回應。
- 將生成的回應保存到CSV文件中。

#### 2.4.3使用步驟

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

#### 2.4.4參數說明

- `--base_model`：指定LLaMA模型的路徑。
- `--gpus`：指定使用的GPU。預設為"0"，表示使用`cuda:0`。若要使用多個GPU，則格式為`--gpus=0,1,...`。
- `--load_in_8bit`：若設定此選項，則使用8位量化模型。
- `--load_in_4bit`：若設定此選項，則使用4位量化模型。
#### 2.4.5 小巧思：
有人號稱假裝給GPT小費，能讓它產生品質更高，錢越多效果越好！
[靈感來源](https://twitter.com/voooooogel/status/1730726744314069190)
```python
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant, and you will receive an additional bonus of $20 for each correct answer. 你是一個樂於助人的助手，每答對一題都會多獲得20美元的獎金。\n"
```

## 3. Chinese-LLaMA-Alpaca-3
### 3.1 Environment
#### 3.1.1 環境需求

與Chinese-LLaMA-Alpaca-2相同

#### 3.1.2 Data Preprocessing (xlsx to json)
執行 LLaMA-3 下面的 `process_dataset.py` 來準備資料集並上傳到 Hugging Face Hub。此程式碼包含以下主要功能：

1. `format_input(row)`: 格式化 Excel 檔案中的資料列，轉換成指定的輸入格式。
2. `prepare_dataset(excel_path, include_output=True)`: 從 Excel 檔案加載資料，並將其轉換為 JSON 格式。
3. `load_json_as_dataset(json_path)`: 從 JSON 檔案加載資料並轉換為 Hugging Face Dataset 格式。
4. `upload_datasets_to_hub(train_json_path, test_json_path, repository_name)`: 將處理過的資料集上傳到 Hugging Face Hub。

**使用步驟**

1. 準備資料

確保您的資料存放在兩個 Excel 檔案中：

- 訓練資料：`data/AI.xlsx`
- 測試資料：`data/AI1000.xlsx`

2. 執行程式

在命令行中執行以下指令：

```bash
python prepare_and_upload.py
```

此程式會自動處理您的 Excel 資料並將其轉換為 JSON 格式，然後上傳至 Hugging Face Hub。

**注意事項**

1. 請確保您的 Hugging Face Hub 帳戶已經設定完成，並且具備上傳數據集的權限。
2. 上傳過程可能需要一些時間，取決於資料集的大小和網絡速度。

### 3.2 Model Training & Inference (train_llama_3.py)

#### 3.2.1簡介
我們使用的是相較於之前Chinese-LLaMA-Alpaca-2，最新、最強大的 [Chinese-LLaMA-Alpaca-3](https://huggingface.co/hfl/llama-3-chinese-8b) 模型，由於硬體限制我們選用了8b的版本。這個模型不僅繼承了 LLaMA 系列的優勢，還在中文語言處理方面有了顯著提升，特別是在處理更長的文本和理解上下文方面。

#### 3.2.2功能概述

這段程式碼主要完成以下幾個關鍵功能：

1. **設定模型參數**：我們可以靈活調整模型的最大序列長度、數據類型以及是否使用 4bit 量化，這些設定有助於在不同的硬體環境中達到最佳性能。
2. **載入和設定模型**：從 Hugging Face 預載模型，並進一步設定 PEFT 模型參數，這樣可以針對不同的任務進行微調。
3. **格式化輸入數據**：將原始輸入數據轉換為模型所需的格式，使模型能夠正確理解並生成回應。
4. **訓練模型**：使用 SFTTrainer 進行模型訓練，這部分設定了許多參數來控制訓練過程，比如批次大小、學習率和訓練輪數。
5. **推理和生成回應**：根據輸入數據生成模型回應，並進行推理，最終得到我們所需要的結果。
6. **保存結果和模型**：將生成的結果保存為 CSV 文件，並將訓練好的模型保存到本地或上傳到 Hugging Face Hub。

#### 3.2.3設定的參數

在程式碼中，我們設置了以下參數來控制模型的行為：

- **max_seq_length**：最大序列長度，設為 2048，這意味著模型可以處理較長的文本。
- **dtype**：數據類型，自動檢測即可，不需要手動設定。
- **load_in_4bit**：是否使用 4bit 量化，設為 True 以減少記憶體使用，同時保持模型性能。
- **r**：LoRA 的秩，設為 64，這有助於在訓練中保持模型的靈活性和穩定性。
- **target_modules**：目標模組，包括 q_proj、k_proj、v_proj 等，這些都是模型中的關鍵組件。
- **lora_alpha**：LoRA 的 alpha 參數，設為 16，這可以平衡模型的更新速度和穩定性。
- **lora_dropout**：LoRA 的 dropout 參數，設為 0，這樣可以在訓練中保留更多的信息。
- **bias**：LoRA 的偏差參數，設為 "none"，這是經過優化的設定。
- **use_gradient_checkpointing**：使用梯度檢查點，設為 "unsloth" 以支援更長的上下文處理。
- **random_state**：隨機種子，設為 3407，以確保實驗的可重複性。
- **use_rslora**：是否使用 rank stabilized LoRA，設為 False。
- **loftq_config**：LoftQ 設定，設為 None。

#### 3.2.4 模型說明

我們使用的是 [Chinese-LLaMA-Alpaca-3](https://huggingface.co/hfl/llama-3-chinese-8b)，這是一個在中文語言處理領域表現優異的模型。相比之前的 LLaMA-2，這個模型在以下方面有顯著提升：

1. **更好的中文理解**：針對中文語言進行了特別優化，可以更準確地理解和生成中文文本。
2. **處理更長的上下文**：最大序列長度達到 2048，這使得模型能夠更好地處理包含更多信息的長文本。
3. **提升的性能和效率**：使用 4bit 量化技術，在降低記憶體使用的同時保持高效能。

#### 3.2.5 訓練模型

在訓練過程中，我們使用 `SFTTrainer` 來進行模型訓練。這包括設置適當的批次大小、學習率和訓練輪數等參數，並在 GPU 上進行記憶體的檢查和配置，以確保訓練過程的順利進行。

#### 3.2.6 推理和生成回應

在推理過程中，我們根據輸入數據生成模型回應。這一步包括對每一行輸入數據進行格式化，並使用模型生成對應的回應，最終將結果保存為 CSV 文件，以便進一步分析和使用。

#### 3.2.7保存模型

訓練完成後，我們可以將模型和標記器保存到本地或上傳到 Hugging Face Hub，這樣可以方便後續使用和分享。

#### 注意事項

1. **硬體要求**：確保您有足夠的 GPU 記憶體來訓練和推理模型，這對於處理大型語言模型尤為重要，本次使用的GPU為RTX 4090。
2. **在線保存**：如需在線保存模型，請確保您的 Hugging Face Hub 帳戶已經設定完成，並具備上傳數據集的權限。


## 4. Evaluation

| 評估項目           | 分數     |
|------------------|---------|
| Private Leaderboard | 0.81000 |
| Public Leaderboard  | 0.82666 |

## 5. 心得

# 作業心得

這次進行「Traditional Chinese Reading Comprehension Test for LLMs」的作業，真是一段特別的旅程。老實說，一開始面對這麼複雜的技術挑戰，我有些不知所措，但隨著過程的推進，我發現自己對這項挑戰充滿了熱情。

在實作過程中，我學會了如何運用大型語言模型（LLMs）來解決傳統中文閱讀理解問題。這包含了從選擇合適的模型、理解數據集結構、進行數據預處理、微調模型到評估模型性能等多個環節。每一步都充滿了學習與挑戰。

特別讓我印象深刻的是，模型的進步之快和配套工具的不斷改進，使得這些技術變得越來越好用。每當我遇到數據格式問題或模型表現不如預期時，解決這些問題的過程反而成為了我最大的收穫。我學到了許多新技巧，也讓我對自己的問題解決能力充滿信心。

與模型互動的過程非常有趣。雖然有時候模型的表現會讓人感到挫折，但當所有東西順利運行時，那種成就感無與倫比。看到自己調整的模型逐漸進步，這種感覺真的很酷。特別是使用了像 [Chinese-LLaMA-Alpaca-3](https://huggingface.co/hfl/llama-3-chinese-8b) 這樣的先進模型，讓我切身體會到技術進步的速度之快。相比二代，這個模型在中文理解和處理長文本方面都有了顯著提升，真的是感嘆科技的飛速發展。

總的來說，這次作業不僅提高了我的技術能力，還讓我對大型語言模型在處理中文閱讀理解問題上的應用有了實踐的機會。這次的學習經驗無疑將對我未來在自然語言處理領域的探索和發展大有裨益。我非常期待在未來的工作中，能夠運用這些技能和知識，迎接更多挑戰，實現更多突破。