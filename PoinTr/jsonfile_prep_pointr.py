import json
import os
import shutil

working_dir = "/home/aidana/Documents/PointTr_data_halved"
directory_path_partial = working_dir+"/partial"
directory_path_complete = working_dir+"/complete"

files = os.listdir(directory_path_complete)

for fold in range(10):
    print("Fold: ", fold+1)
    train_files = []
    val_files = []
    for file in files:

        if "Specimen_{}_".format(fold+1) in file:
            val_files.append(file[:-4])
        else:
            train_files.append(file[:-4])


    data = {
        "taxonomy_id": "10102023",
        "taxonomy_name": "vertebrae",
        "train": train_files,
        "val": val_files

    }

    print(len(val_files))
    print(len(train_files))

    with open(working_dir + '/fold_{}.json'.format(fold+1), 'w') as json_file:
        json.dump([data], json_file, indent=4)