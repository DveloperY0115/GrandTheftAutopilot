import os
import cv2
import numpy as np
import datetime
# import Autopilot.Perception.ImageGrab as ImageGrab
from PIL import ImageGrab, Image
import win32api as wapi
import pandas as pd
import os

from core.img_process import Image_Processor

# 800 * 600


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def capture_image():
    drive_view = ImgProc.grab_screen()
    mapview = drive_view[480:590, 5:160]
    direction = drive_view[570:580, 15:25]
    # self.drive_view = img
    # Preprocess needed
    return drive_view, mapview, direction


class DataCollect:
    # drive_view_bbox = (0, 40, 800, 640)
    mapview_bbox = (5, 480, 160, 590)
    direction_bbox = (15, 570, 25, 580)

    def __init__(self):
        self.keyList = ["\b"]
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789,.'£$/\\":
            self.keyList.append(char)
        self.target_folder = './dataset/' + datetime.datetime.utcnow().strftime(
            "%y%m%d_%H-%M-%S") + "_" + "data/"
        try:
            if not os.path.exists(self.target_folder):
                os.makedirs(self.target_folder)
        except OSError:
            print('Error: Creating directory. ' + self.target_folder)

    def capture_key(self):
        for button in self.keyList:
            if wapi.GetAsyncKeyState(ord(button)):
                if button == 'A':
                    return 'A'
                elif button == 'D':
                    return 'D'
                elif button == 'X':
                    return 'X'
                elif button == 'Q':
                    return 'Q'
                else:
                    return
        return

    # save format: date_key (key is one of w,a,s,d)
    # for example, 2020-11-17-20:50:34_w
    def save_data(self, drive_view_img, mapview_img, direction_img, control):
        # if control == 'w' or control == 'd': continue
        # save captured images in 'Autopilot/dataset/imgs/(file names)''
        target_directory = self.target_folder + 'imgs/'
        drive_view_filename = target_directory + 'drive_view/' + datetime.datetime.utcnow().strftime(
            "%y%m%d_%H-%M-%S") + "_" + "drive_view" + '.jpg'  # numpy array
        mapview_filename = target_directory + 'mapview/' + datetime.datetime.utcnow().strftime(
            "%y%m%d_%H-%M-%S") + "_" + "mapview" + '.jpg'  # numpy array
        direction_filename = target_directory + 'direction/' + datetime.datetime.utcnow().strftime(
            "%y%m%d_%H-%M-%S") + "_" + "direction" + '.jpg'  # numpy array

        cv2.imwrite(drive_view_filename, drive_view_img)
        cv2.imwrite(mapview_filename, mapview_img)
        cv2.imwrite(direction_filename, direction_img)

        if control is None:
            control = "None"

        temp_dict = {'drive_view': drive_view_filename, 'mapview': mapview_filename,
                     'direction': direction_filename, 'control': control, 'speed': 0}
        return temp_dict


if __name__ == '__main__':
    index = 0
    start_flag = False
    data_dict = {}
    dc = DataCollect()
    ImgProc = Image_Processor()
    while True:
        key_input = dc.capture_key()
        if key_input == 'X':
            start_flag = True

        if not start_flag:
            continue

        # Capture All images and Change them to RGB
        # drive_view, mapview, direction = capture_image()
        # drive_view = cv2.cvtColor(drive_view, cv2.COLOR_BGR2RGB)
        # mapview = cv2.cvtColor(mapview, cv2.COLOR_BGR2RGB)
        # direction = cv2.cvtColor(direction, cv2.COLOR_BGR2RGB)
        drive_view = ImgProc.grab_screen()

        cv2.imshow("Drive_view", drive_view)
        # cv2.imshow("Mapview", mapview)
        # cv2.imshow("Direction", direction)

        # data_log = dc.save_data(drive_view, mapview, direction, key_input)
        # print(data_log)
        #
        # index += 1
        # data_dict[index] = data_log
        #
        # # When Q is pressed, all data are organized into a csv file.
        # # Then, program is exited.
        # if key_input == "Q":
        #     cv2.destroyAllWindows()
        #     dataset = pd.DataFrame.from_dict(data_dict, orient='index')
        #     dataset_name = dc.target_folder + datetime.datetime.utcnow().strftime(
        #         "%y%m%d_%H-%M-%S") + "_dataset.csv"
        #     dataset.to_csv(dataset_name)
        #     break

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            dataset = pd.DataFrame.from_dict(data_dict, orient='index')
            dataset_name = dc.target_folder + datetime.datetime.utcnow().strftime(
                "%y%m%d_%H-%M-%S") + "_dataset.csv"
            dataset.to_csv(dataset_name)
            break
