# TODO list
# 1. Encode W, A, S, D, NaN to numbers
# 2. Make Generator
# 3. Preprocess images (Define ROI)
# 4. Normalize images

import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import keras
import tensorflow as tf

from keras.models import Sequential
from keras.models import load_model

from keras.layers import Flatten, Dense
from keras.layers import BatchNormalization
from keras.layers import Conv2D

from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint

# One-hot encoder
def control2num(control):
    if control == NaN:
        pass
    elif control == "D":
        pass
    elif control == "D":
        pass
    elif control == "D":
        pass
    else:
        pass

    return

def num2control(num):

    return

def img_to_arr(p):
    with image.load_img(p) as img:
        img = image.img_to_array(img)
    return img

# values computed from dataset sample.
def normalize(img):
    img[:,:,0] -= 94.9449
    img[:,:,0] /= 58.6121

    img[:,:,1] -= 103.599
    img[:,:,1] /= 61.6239

    img[:,:,2] -= 92.9077
    img[:,:,2] /= 68.66

    return img

# define generator that loops through the data
def generator(df_path, img_shape):
    """
    A function which creates arrays of images and labels out of dataframe.

    @inputs
    df_path: (Absolute or Relative) path to CSV files containing info of images and labels
    img_shape: Shape of image (ex. (width, height, channels))
    @returns
    a tuple of form ((array of images), (array of labels))
    """
    # shuffle dataframe for each epoch
    loader = DataLoader()
    df = loader.load_csv(path)

    # lists for holding data
    img_list = []
    control_list = []

    img_series = df['drive_view']
    control_series = df['control']

    for i in range(len(img_series)):
        cur_img = loader.load_img(df['drive_view'].iloc[i])
        cur_img = tf.image.per_image_standardization(cur_img)
        img_list.append(cur_img)

    for j in range(len(control_series)):
        control_list.append(str(df['control'].iloc[j]))

    return (np.array(img_list), np.array(control_list))
    # idea
    # csv file
    # [drive view] ... [control]
    # (data) ... (data)
    # ...         ...

    # img_list = [(loaded image (3d), images...] --> (num_samples, width, height, channels)
    # input_list = ["W", "A", ... "None"] --> One Hot Encoding --> (num_samples, )

    # replace batch_size with len(img_list)?
    # create empty batch
    """
    batch_img = np.zeros((batch_size,) + img_shape)
    batch_label = np.zeros((batch_size, 1))

    OUTPUT_NORMALIZATION = 4 # constant to make output bounded

    index = 0
    while True:
        for i in range(batch_size):
            img_name = img_list[index]
            arr = loader.load_img(index)

            # batch_img[i] = normalize(arr)
            batch_label[i] = controls[index]
            index += 1
            if index == len(img_list):
                index = 0

        yield batch_img, batch_label
    """


## 임시로 옮겨 놓은 거 ##
from sklearn.preprocessing import LabelBinarizer
import numpy as np
train_y = np.array(['W', 'S', 'D', 'A', ''])
binarizer = LabelBinarizer()
lb = binarizer.fit_transform(train_y)
train_y = lb.transform(train_y)

#### 구분선 ####

def normalize
