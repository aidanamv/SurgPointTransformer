import re
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Define the path to your .txt file
file_path = r"G:\nll_loss_based_ckpts\fold_1.out"

# Regular expressions to extract relevant lines and information
train_regex = re.compile(r'\[\d+: (\d+)/\d+\] train loss: [\d.]+ accuracy: ([\d.]+)')
test_regex = re.compile(r'\[\d+: (\d+)/\d+\] \x1b\[94mtest\x1b\[0m loss: [\d.]+ accuracy: ([\d.]+)')


# Lists to store the extracted accuracies and epoch numbers
train_accuracies = []
test_accuracies = []
train_accuracies_all = []

train_epoch = []
train_epoch_all = []

test_epoch = []

# Read the file
with open(file_path, 'r') as file:
    data = file.read()


# Process each line in the file
for line in data.splitlines():
    train_match = train_regex.match(line)
    if train_match:
        epoch = train_match.group(0)
        accuracy = float(train_match.group(2))
        train_accuracies_all.append(accuracy)
        train_epoch_all.append(epoch)

    test_match = test_regex.match(line)
    if test_match:
        epoch =  test_match.group(0)
        accuracy = float(test_match.group(2))
        test_accuracies.append(accuracy)
        test_epoch.append(epoch)
        train_accuracies.append(train_accuracies_all[-1])
        train_epoch.append(train_epoch_all[-1])



print(len(test_accuracies))
print(np.max(test_accuracies))
ind =np.where(test_accuracies ==np.max(test_accuracies))
print(ind)
print(test_epoch[int(ind[0])])
plt.plot(test_accuracies)
plt.show()