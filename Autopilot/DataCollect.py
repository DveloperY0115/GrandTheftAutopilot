import os
import cv2
import numpy as np
import datetime
# import Autopilot.Perception.ImageGrab as ImageGrab
from PIL import ImageGrab, Image
import win32api as wapi
import pandas as pd

from core.img_process import *
ImgProc = Image_Processor

# 800 * 600


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class DataCollect:
    #### Global Variables ####
    # frontview_bbox = (0, 40, 800, 640)
    mapview_bbox = (5, 480, 160, 590)
    direction_bbox = (15, 570, 25, 580)

    def __init__(self):
        self.keyList = ["\b"]
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789,.'£$/\\":
            keyList.append(char)

    def capture_key(self):
        for key in self.keyList:
            if wapi.GetAsyncKeyState(ord(key)):
                if key == 'A':
                    return 'A'
                elif key == 'D':
                    return 'D'
                else:
                    return
        return


    def capture_frontview(self):
        img = ImgProc.grab_screen()
        # Preprocess needed
        return img


    def capture_mapview(self, frontview):
        img = frontview[480:590, 5:160]
        return img


    def capture_direction(self, frontview):
        img = frontview[570:580, 15:25]
        return img


    # save format: date_key (key is one of w,a,s,d)
    # for example, 2020-11-17-20:50:34_w
    def save_data(self, frontview_img, mapview_img, direction_img, control):
        # if control == 'w' or control == 'd': continue
        # save captured images in 'Autopilot/dataset/imgs/(file names)''
        target_folder = './dataset' + datetime.datetime.utcnow().strftime(
            "%y%m%d_%H-%M-%S") + "_" + "data/"
        target_directory = './dataset/' + target_foldername + 'imgs/'
        frontview_filename = target_directory + 'frontview/' + datetime.datetime.utcnow().strftime(
            "%y%m%d_%H-%M-%S") + "_" + "frontview" + '.jpg'  # numpy array
        mapview_filename = target_directory + 'mapview/' + datetime.datetime.utcnow().strftime(
            "%y%m%d_%H-%M-%S") + "_" + "mapview" + '.jpg'  # numpy array
        direction_filename = target_directory + 'direction/' + datetime.datetime.utcnow().strftime(
            "%y%m%d_%H-%M-%S") + "_" + "direction" + '.jpg'  # numpy array

        cv2.imwrite(frontview_filename, frontview_img)
        cv2.imwrite(mapview_filename, mapview_img)
        cv2.imwrite(direction_filename, direction_img)

        if control == None:
            control = "None"

        temp_dict = {'frontview': frontview_filename, 'mapview': mapview_filename,
                     'direction': direction_filename, 'control': control, 'speed': 0}
        return temp_dict


if __name__ == '__main__':
    index = 0
    start_flag = False
    data_dict = {}
    while True:
        keyinput = capture_key()
        if keyinput == "X":
            start_flag = True

        if start_flag == False:
            continue

        # Capture All images and Change them to RGB
        frontview = capture_frontview()
        mapview = capture_mapview(frontview)
        direction = capture_direction(frontview)

        # frontview = cv2.cvtColor(frontview, cv2.COLOR_BGR2RGB)
        # mapview = cv2.cvtColor(mapview, cv2.COLOR_BGR2RGB)
        # direction = cv2.cvtColor(direction, cv2.COLOR_BGR2RGB)

        cv2.imshow("Frontview", frontview)
        cv2.imshow("Mapview", mapview)
        cv2.imshow("Direction", direction)

        data_log = save_data(frontview, mapview, direction, keyinput)
        print(data_log)

        index += 1
        data_dict[index] = data_log

        key = cv2.waitKey(1) & 0xFF

        # When Q is pressed, all data are organized into a csv file.
        # Then, program is exited.
        if key == ord("q"):
            cv2.destroyAllWindows()
            dataset = pd.DataFrame.from_dict(data_dict, orient='index')
            dataset_name = target_folder + target_directorydatetime.datetime.utcnow().strftime("%y%m%d_%H-%M-%S")+"_dataset.csv"
            dataset.to_csv(dataset_name)
            break
