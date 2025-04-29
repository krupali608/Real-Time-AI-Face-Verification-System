import os
import random
import shutil
from itertools import islice

outputFolderPath = "Anti-Spoofing/Dataset/splitData"
inputFolderPath = "Anti-Spoofing/Dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

try:
    shutil.rmtree(outputFolderPath)
except OSError as e:
    os.mkdir(outputFolderPath) 
    
# Directories to create
os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)

# get the names
listNames = os.listdir(inputFolderPath)

uniqueNames = set()  # Using a set to prevent duplicates

for name in listNames:
    if name.startswith('.'):  # Ignore hidden files
        continue
    if '.' in name:  # Ensure the filename contains an extension
        base_name, ext = os.path.splitext(name)  # Extract base name
        if base_name.strip():  # Ensure it's not empty
            uniqueNames.add(base_name)

uniqueNames = list(uniqueNames)  # Convert back to a list if needed
# print(len(uniqueNames))

# shuffle
random.shuffle(uniqueNames)

# find number of images for each folder
lenData = len(uniqueNames)
lenTrain = int(lenData*splitRatio['train'])
lenVal = int(lenData*splitRatio['val'])
lenTest = int(lenData*splitRatio['test'])

# put remaining images in training
if lenData != lenTrain+lenTest+lenVal:
    remaining = lenData-(lenTrain+lenTest+lenVal)
    lenTrain += remaining


# split the list
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images:{lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')
#print(Output)
#print(len(Output))

# copy the files
sequence = ['train', 'val', 'test']
for i,out in enumerate(Output):
    for fileName in out:
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputFolderPath}/{fileName}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')
print("Split Process Completed...")


# creating data.yaml file
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc:{len(classes)}\n\
names: {classes}'

f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yml file Created...")
