import os
import cv2
import datetime
import win32api as wapi
import numpy as np
import pandas as pd
import os
from utils import resize

from core.img_process import Image_Processor

# 800 * 600
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class DataCollect:
    def __init__(self):
        self.keyList = ["\b"]
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789,.'£$/\\":
            self.keyList.append(char)
        self.target_folder = './dataset/' + datetime.datetime.utcnow().strftime(
            "%y%m%d_%H-%M-%S") + "_" + "data/"
        # Set all the folders
        try:
            if not os.path.exists(self.target_folder):
                os.makedirs(self.target_folder)
            if not os.path.exists(self.target_folder + "imgs"):
                os.makedirs(self.target_folder + "imgs")

        except OSError:
            print('Error: Creating directory.')

    # Capture Key
    # A, D: Represents steering angle in GTA
    # S, E: Represents Start and End.
    def capture_key(self):
        for button in self.keyList:
            if wapi.GetAsyncKeyState(ord(button)):
                if button == 'A':
                    return 'A'
                elif button == 'D':
                    return 'D'
                elif button == 'W':
                    return 'W'
                elif button == 'S':
                    return 'S'
                elif button == 'E':
                    return 'E'
                else:
                    return
        return

    """
    def save_data(self, drive_view_img, mapview_img, direction_img, control, index):
        # if control == 'w' or control == 'd': continue
        # save captured images in 'Autopilot/dataset/imgs/(file names)''
        target_filename = self.target_folder + 'imgs/' + "drive_view" + str(index) + '.jpg'  # numpy array
        # # + datetime.datetime.utcnow().strftime("%y%m%d_%H-%M-%S")
        drive_view_img = resize(drive_view_img)
        cv2.imwrite(target_filename, drive_view_img)
        # cv2.imwrite(mapview_filename, mapview_img)
        # cv2.imwrite(direction_filename, direction_img)
        temp_dict = {'drive_view': target_filename, 'control': control}
        return temp_dict
    """

    def save_as_numpy(self, img_list, label_list):
        """
        Save the list of images, keys as .npy file in designated path
        :param img_list: List of flattened images
        :param label_list: List of user inputs
        :param path: Path in which the files are saved to
        :return: Nothing
        """
        img_path = self.target_folder + "img.npy"
        label_path = self.target_folder + "label.npy"
        np.save(img_path, np.array(img_list))
        np.save(label_path, np.array(label_list))


if __name__ == '__main__':
    index = 0
    max_sample_num = 10000  # maximum number of samples in a single data
    start_flag = False
    img_list = []    # a list to store flattened images
    label_list = []  # a list to store corresponding input from player
    data_dict = {}

    dc = DataCollect()
    ImgProc = Image_Processor()

    while True:
        """
        Stop collecting data and terminate if,
        (1) User pressed 'q"
        (2) Number of collected sample exceeds 10,000
        """
        if (cv2.waitKey(25) & 0xFF == ord("q")) or (index > max_sample_num - 1):
            cv2.destroyAllWindows()
            # dataset = pd.DataFrame.from_dict(data_dict, orient='index')
            # dataset_name = dc.target_folder + "dataset.csv"
            # dataset.to_csv(dataset_name)
            dc.save_as_numpy(img_list, label_list)
            break

        # TODO: Implement a routine which saves data periodically (might require multithreadiing for efficiency)
        # TODO: Also, if data is thought to be too large, consider using 'HDF5' format instead of '.npy' format

        key_input = dc.capture_key()
        # If press S, data starts to be collected
        # If press E, data collecting stops
        # After pressing E, Select cv2 screen and press Q.
        # Program will save whole csv file and terminate.
        if key_input == 'S':
            start_flag = True
        elif key_input == 'E':
            start_flag = False

        if not start_flag:
            continue

        drive_view = ImgProc.grab_screen()
        img_list.append(drive_view.reshape(1, -1))   # append flattened image to the list
        label_list.append(key_input)    # append user input to the list
        # mapview = drive_view[480:590, 5:160]
        # direction = drive_view[570:580, 15:25]

        cv2.imshow("Drive_view", drive_view)
        # cv2.imshow("Mapview", mapview)
        # cv2.imshow("Direction", direction)

        # data_log = dc.save_data(drive_view, mapview, direction, key_input, index)
        # print(key_input)
        # print(data_log)

        index += 1
        # data_dict[index] = data_log
