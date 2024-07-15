import json
import os
import shutil

directory_path_partial = "/Volumes/SpineDepth/PoinTr_dataset/partial"
directory_path_complete = "/Volumes/SpineDepth/PoinTr_dataset/complete"

file_names = os.listdir(directory_path_partial)

save_dir = "/Users/aidanamassalimova/Documents/FinalDataset_4096"

for i in range(2):
    train_files = []
    test_files = []

    test_directory_partial = save_dir + "/fold_{}/val/partial/10102023".format(i)
    test_directory_complete = save_dir + "/fold_{}/val/complete/10102023".format(i)

    train_directory_complete = save_dir + "/fold_{}/train/complete/10102023".format(i)
    train_directory_partial = save_dir + "/fold_{}/train/partial/10102023".format(i)

    for file in file_names:
        if "._" in file:
            continue
        else:
            if "Specimen_{}_".format(i+1) in file:
                test_files.append(file)
            else:
                train_files.append(file)

    data = {
        "taxonomy_id": "10102023",
        "taxonomy_name": "vertebrae",
        "train": train_files,
        "val": test_files

    }

    print(len(test_files))
    print(len(train_files))


    for file_name in test_files:
        print(file_name)
        source_path1 = os.path.join(directory_path_partial, file_name,"00.pcd")
        if os.path.exists(source_path1):
            destination_path1 = os.path.join(test_directory_partial, file_name)
            if not os.path.exists(destination_path1):
                os.makedirs(destination_path1)
            shutil.copy(source_path1, destination_path1)

        source_path2 = os.path.join(directory_path_complete, file_name+".pcd")
        if os.path.exists(source_path2):
            destination_path2 = os.path.join(test_directory_complete)
            if not os.path.exists(destination_path2):
                os.makedirs(destination_path2)
            shutil.copy(source_path2, destination_path2)



    for file_name in train_files:
        print(file_name)
        source_path3 = os.path.join(directory_path_partial, file_name,"00.pcd")
        if os.path.exists(source_path3):
            destination_path4 = os.path.join(train_directory_partial, file_name)
            if not os.path.exists(destination_path4):
                os.makedirs(destination_path4)
            shutil.copy(source_path3, destination_path4)
        source_path4 = os.path.join(directory_path_complete, file_name+".pcd")
        if os.path.exists(source_path4):
            destination_path5 = os.path.join(train_directory_complete)
            if not os.path.exists(destination_path5):
                os.makedirs(destination_path5)
            shutil.copy(source_path4, destination_path5)



    # Create a JSON file and write the data
    with open(save_dir + '/fold_{}/fold_{}.json'.format(i, i), 'w') as json_file:
        json.dump(data, json_file, indent=4)