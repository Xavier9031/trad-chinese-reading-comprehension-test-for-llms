import pandas as pd
from datasets import Dataset, DatasetDict

def load_json_as_dataset(json_path, include_output):
    # Load JSON line by line and convert to Dataset
    data = pd.read_json(json_path, lines=True)
    
    # Ensure it contains all necessary columns
    if not include_output:
        data['output'] = ""  # Use an empty string to maintain the same data type
    
    return Dataset.from_pandas(data)

def upload_datasets_to_hub(train_json_path, test_json_path, repository_name):
    # Load training dataset (including output)
    train_dataset = load_json_as_dataset(train_json_path, include_output=True)

    # Load testing dataset (originally without output)
    test_dataset = load_json_as_dataset(test_json_path, include_output=False)

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # Upload and save datasets to Hugging Face Hub
    dataset_dict.push_to_hub(repository_name, private=True)

if __name__ == '__main__':
    # Paths to the JSON files and the repository name on Hugging Face Hub
    train_json_path = 'train_dataset.json'
    test_json_path = 'test_dataset.json'
    repository_name = "Xavier9031/ZH-TW_Reading_Comprehension"

    # Print statements for progress tracking
    print("Uploading datasets to the Hub...")
    upload_datasets_to_hub(train_json_path, test_json_path, repository_name)
    print("Datasets uploaded successfully!")
