# Load image data to PilotNet
# csv file consists of image_path and key_input (W,A,S,D)

import sys
import os
import glob

from utils import resize

# csv_test = pd.read_csv("./dataset/")

"""
path_dir = './dataset'
dataset_list = os.listdir(path_dir)
for current_dataset in dataset_list:
    current_dataset_path = path_dir + "/" + current_dataset
    for filename in os.listdir(current_dataset_path):
        if glob.glob(current_dataset_path + "/*.csv"):
            print(filename)
"""

# 여기서 image preprocess

class DataLoader:

    def __init__(self):
        self.data_dir = ".\dataset"

    def load(self):


if __name__ == "__main__":
    path_dir = '.\dataset'
    dataset_list = os.listdir(path_dir)
    for current_dataset in dataset_list:
        current_dataset_path = path_dir + "\\" + current_dataset
        for filename in glob.glob(current_dataset_path + "\*.csv"):
            print(filename)
