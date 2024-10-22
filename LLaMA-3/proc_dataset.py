from datasets import Dataset, DatasetDict, load_dataset
import json

import pandas as pd

def format_input(row):
    options = f"選項1:{str(row['選項1'])}, 選項2:{str(row['選項2'])}, 選項3:{str(row['選項3'])}, 選項4:{str(row['選項4'])}"
    return f"問題:{row['問題']};\n {options};\n 文章:{row['文章']}"

def prepare_dataset(excel_path, include_output=True):
    # Load dataset from an Excel file
    df = pd.read_excel(excel_path)

    # Apply the function to create the 'input' column
    df['input'] = df.apply(format_input, axis=1)

    # All entries in 'instruction' are the same
    df['instruction'] = "請閱讀下面關於問題的描述及參考選項，根據下面提供的文章選出正確的選項，請只用單純的阿拉伯數字回答出正確的選項"

    # Ensure the 'id' is set correctly
    columns = ['id', 'input', 'output', 'instruction'] if include_output else ['id', 'input', 'instruction']

    if include_output:
        df['id'] = df['ID']
        # Set 'output' to the '正確答案' column
        df['output'] = df['正確答案'].astype(str)
    else:
        df['id'] = df['題號']

    # Convert DataFrame to JSON and save
    json_path = 'train_dataset.json' if include_output else 'test_dataset.json'
    df[columns].to_json(json_path, orient='records', lines=True, force_ascii=False)

def load_json_as_dataset(json_path):
    # 加载 JSON 文件并转换为 Dataset
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return Dataset.from_pandas(pd.DataFrame(data))

def upload_datasets_to_hub(train_json_path, test_json_path, repository_name):
    # 加载训练和测试数据集
    train_dataset = load_json_as_dataset(train_json_path)
    test_dataset = load_json_as_dataset(test_json_path)

    # 创建 DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # 上傲和保存数据集到 Hugging Face Hub
    dataset_dict.push_to_hub(repository_name, private=True)


if __name__ == '__main__':
    # Paths to the Excel files
    train_excel_path = 'data/AI.xlsx'
    test_excel_path = 'data/AI1000.xlsx'

    # 指定 JSON 文件路径
    train_json_path = 'train_dataset.json'
    test_json_path = 'test_dataset.json'
    repository_name = "Xavier9031/ZH-TW_Reading_Comprehension"


    # # Process and save the datasets
    print("Processing and saving datasets...")
    prepare_dataset(train_excel_path, include_output=True)  # For training data
    prepare_dataset(test_excel_path, include_output=False)  # For testing data
    print("Datasets saved successfully!")
