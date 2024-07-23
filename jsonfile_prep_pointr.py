import json
import os
import shutil
fold = 7
directory_path_partial = "./PoinTr/data/PCN/fold_{}/partial".format(fold)
directory_path_complete = "./PoinTr/data/PCN/fold_{}/complete".format(fold)

dir = "./PoinTr/data/PCN/fold_{}/".format(fold)

train_files = os.listdir(os.path.join(dir, "train","partial","10102023"))
val_files  = os.listdir(os.path.join(dir, "val","partial","10102023"))


data = {
    "taxonomy_id": "10102023",
    "taxonomy_name": "vertebrae",
    "train": train_files,
    "val": val_files

}

print(len(val_files))
print(len(train_files))






# Create a JSON file and write the data
with open(dir + '/fold_{}.json'.format(fold), 'w') as json_file:
    json.dump([data], json_file, indent=4)