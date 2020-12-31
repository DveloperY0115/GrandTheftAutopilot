import os
import cv2
import numpy as np
import datetime
# import Autopilot.Perception.ImageGrab as ImageGrab
from PIL import ImageGrab
# 1280 * 720
radar_bbox = (20,600,200,720)
direction_bbox = (30,710,42,720)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

img_w, img_h = 800,600

import win32api as wapi

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789,.'£$/\\":
    keyList.append(char)

def capture_key():
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            if key == 'A':
                return 'A'
            elif key =='D':
                return 'D'
            else:
                return
    return

def capture_frontview():
    img = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    return img

def capture_mapview():
    img = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    return img


# save format: date_key (key is one of w,a,s,d)
# for example, 2020-11-17-20:50:34_w
def save_data(data_img, control):
    # if control == 'w' or control == 'd': continue
    # save captured images in 'Autopilot/dataset/imgs/(file names)''
    target_directory = 'dataset/imgs'
    file_path =  datetime.datetime.utcnow().strftime("%y-%m-%d:%H:%M:%S") + "_" + control + '.jpg' # numpy array
    cv2.imwrite(file_path, data_img)


def main():
    while True:
        frontview = capture_frontview()
        # mapview = capture_mapview()
        key = capture_key()
        print(frontview + " " + key)
        #img = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        # cv2.imshow("Frame", original_img)
        # key = cv2.waitKey(1) & 0xFF
        # save_data(img, chr(key)) # lower case alphabet
        # if key == ord("q"):
        #     cv2.destroyAllWindows()
        #     break
