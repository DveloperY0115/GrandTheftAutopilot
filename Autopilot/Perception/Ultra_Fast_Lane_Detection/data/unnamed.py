import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos


def loader_func(path):
    return Image.open(path)


path = 'C:/Users/NA/Desktop/cuLane/cuLane_dataset/'
list_path = 'C:/Users/NA/Desktop/cuLane/cuLane_dataset/list/test_split/test8_night.txt'
with open(list_path, 'r') as f:
    list = f.readlines()
list = [l[1:] if l[0] == '/' else l for l in list]  # exclude the incorrect path prefix '/' of CULane
with open('C:/Users/NA/Desktop/cuLane/cuLane_dataset/list/test_split/new_test8_night.txt', 'w') as f:
    for fpath in list:
        img_path = os.path.join(path, fpath)
        try:
            img = loader_func(img_path.split()[0])
        except:
            continue
        f.write(fpath)
