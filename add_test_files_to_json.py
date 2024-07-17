import json
import os

dir = r"F:\PoinTr_dataset\FinalDataset_4096\fold_1\fold_1.json"
# Open and read the JSON file
with open(dir, 'r') as file:
    data = json.load(file)

# Print the contents of the JSON file
print(data)

test_files = os.listdir(r"F:\PoinTr_dataset\FinalDataset_4096\fold_1\test\10102023\partial")

data[0]["test"] = test_files

with open(dir, 'w') as json_file:
    json.dump(data, json_file, indent=4)