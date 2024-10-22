import pandas as pd
import argparse 

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default='AI.xlsx')
parser.add_argument('--output_path', type=str, default='AI.json')
parser.add_argument('--instruction', type=str, default='請根據以下輸入內容詳細思考後，回答選擇題，並只以阿拉伯數字回答：\n')
parser.add_argument('--sample_rate', type=float, default=1)
args = parser.parse_args()


# Load the dataset
file_path = args.file_path
data = pd.read_excel(file_path)

# Randomly sample the data
if args.sample_rate < 1:
    data = data.sample(frac=args.sample_rate ,random_state=0)

# Preparing the data in the specified JSON format
json_data = []

for index, row in data.iterrows():
    # Combine the article and question
    input_text = str(row['文章']) + "\n問題：" + str(row['問題']) + "\n選項1：" + str(row['選項1']) + "\n選項2：" + str(row['選項2']) + "\n選項3：" + str(row['選項3']) + "\n選項4：" + str(row['選項4'])

    # Format into the desired structure
    json_entry = {
        "instruction": args.instruction,
        "input": input_text,
        "output": str(row['正確答案'])
    }
    json_data.append(json_entry)

# Convert the list to JSON
import json
json_output = json.dumps(json_data, ensure_ascii=False, indent=4)

# Save the JSON file
with open(args.output_path, 'w') as outfile:
    outfile.write(json_output)


