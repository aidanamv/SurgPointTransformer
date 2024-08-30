import os
import shutil

# Function to create directories if they don't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)





imgs_data_dir = "/media/aidana/SpineDepth/YOLO/images"
labels_data_dir = "/media/aidana/SpineDepth/YOLO/labels"

files = os.listdir(imgs_data_dir)



for i in range(1,10):
    train_files = []
    test_files = []

    # Paths to your dataset
    train_dir = "/home/aidana/SpineDepth/YOLO/fold_{}/train".format(i+1)
    test_dir = "/home/aidana/SpineDepth/YOLO/fold_{}/test".format(i+1)


    print("processing fold {}".format(i+1))

    for file in files:
        if "._" in file:
            continue
        if "Specimen_{}_".format(i + 1) in file:
            test_files.append(file)
        else:
            train_files.append(file)
    # Copy files to the respective directories
    for file in train_files:
        create_dir(os.path.join(train_dir, "images"))
        create_dir(os.path.join(train_dir, "labels"))
        shutil.copy(os.path.join(imgs_data_dir, file), os.path.join(train_dir, "images",file))
        shutil.copy(os.path.join(labels_data_dir, file.replace("png","txt")), os.path.join(train_dir, "labels",file.replace("png","txt")))


    for file in test_files:
        create_dir(os.path.join(test_dir, "images"))
        create_dir(os.path.join(test_dir, "labels"))
        shutil.copy(os.path.join(imgs_data_dir, file), os.path.join(test_dir, "images",file))
        shutil.copy(os.path.join(labels_data_dir, file.replace("png","txt")), os.path.join(test_dir, "labels",file.replace("png","txt")))

    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")




