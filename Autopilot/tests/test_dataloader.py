import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from DataLoader import DataLoader

# define generator that loops through the data

def generator(df_path):
    """
    A function which creates arrays of images and labels out of dataframe.

    :param df_path: (Absolute or Relative) path to CSV files containing info of images and labels
    :param img_shape: Shape of image (ex. (width, height, channels))
    :return: a tuple of form ((array of images), (array of labels))
    """
    # initialize loader and load CSV file
    loader = DataLoader('../dataset', use_colab=False)
    df = loader.load_csv(df_path)

    # lists for holding data
    img_list = []
    control_list = []

    img_series = df['drive_view']
    control_series = df['control']

    print("Total number of images:", len(img_series))
    print("Total number of labels:", len(control_series))

    for i in range(len(img_series)):
        if i % 100 == 0:
           print("Processing", i, "th image")
        cur_img = loader.load_img(i)
        img_list.append(cur_img)

    for j in range(len(control_series)):
        if j % 100 == 0:
          print("Processing", j, "th label")
        control_list.append(str(df['control'].iloc[j]))

    assert len(img_list) == len(control_list), "Length of two lists are not equal"

    return np.array(img_list), np.array(control_list)


if __name__ == "__main__":
    path_dir = '../dataset'

    loader = DataLoader("../dataset", use_colab=False)
    X, y = loader.load_data("210125_04-18-29_data")

    print("Number of images:", len(X))
    print("Number of labels:", len(y))

    test_img = X[11].reshape(594, 800, 3)
    while True:
        cv2.imshow("img", test_img)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
            

    """
    X, y = generator('210125_04-18-29_data')
    """
