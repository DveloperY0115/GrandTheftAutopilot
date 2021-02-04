# Load image data to PilotNet
# csv file consists of image_path and key_input (W,A,S,D)

import sys
import os
import glob
import cv2
import pandas as pd
import numpy as np

from utils import resize

# csv_test = pd.read_csv("./dataset/")

""""
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

    def __init__(self, path_to_datasets, use_colab=False):
        if not use_colab:
            # train neural network in local machine
            self.path_to_datasets = path_to_datasets
        else:
            # assumes that the data is mounted on Google Drive
            self.path_to_datasets = "/content/drive/MyDrive/AutopilotDrive/dataset"
        return

    def load_data(self, data_dir_name):
        full_path = self.path_to_datasets + "/" + data_dir_name
        print(full_path)
        for file in glob.glob(full_path + "/" + "img.npy"):
            img = np.load(file, allow_pickle=True)
        for file in glob.glob(full_path + "/" + "label.npy"):
            label = np.load(file, allow_pickle=True)
            return (img, label)


    def load_csv(self, data_dir_name):
        self.full_path = self.path_to_datasets + "/" + data_dir_name
        for file in glob.glob(self.full_path + "/" + "*.csv"):
            # There should be only one CSV file in the given directory
            df = pd.read_csv(file)
            self.df = df
            return df

    def load_img(self, sample_num):
        # TODO: We probably have to change the way path to image file is saved
        img_path = self.path_to_datasets + "/../" + str(self.df["drive_view"].iloc[sample_num])
        img = cv2.imread(img_path)
        return img
