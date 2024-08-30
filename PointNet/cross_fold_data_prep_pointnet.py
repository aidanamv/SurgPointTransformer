import os
import json

dir = "/Volumes/SpineDepth/PointNet_data/1"
files = os.listdir(dir)

for i in range(10):
    train_files = []
    val_files = []
    print("processing fold {}".format(i+1))

    for file in files:
        if "._" in file:
            continue
        if "Specimen_{}_".format(i + 1) in file:
            val_files.append("PointNet_data/1/" +file[:-4])
        else:
            train_files.append("PointNet_data/1/" +file[:-4])

    save_dir =os.path.join( "/Volumes/SpineDepth/PointNet_data", "fold_{}".format(i + 1))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    with open(os.path.join(save_dir,'train_data.json'), 'w') as f:
        json.dump(train_files, f)
    with open(os.path.join(save_dir,'val_data.json'), 'w') as f:
        json.dump(val_files, f)

