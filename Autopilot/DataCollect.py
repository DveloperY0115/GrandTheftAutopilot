import os
import cv2
import numpy as np
import datetime
# import Autopilot.Perception.ImageGrab as ImageGrab
from PIL import ImageGrab
import win32api as wapi

# 1280 * 720
map_bbox = (5, 520, 160, 630)
direction_bbox = (15, 610, 25, 620)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

img_w, img_h = 800, 600

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789,.'£$/\\":
    keyList.append(char)


def capture_key():
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            if key == 'A':
                return 'A'
            elif key == 'D':
                return 'D'
            else:
                return
    return


def capture_frontview():
    img = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    # Preprocess needed
    return img


def capture_map():
    img = np.array(ImageGrab.grab(bbox=map_bbox))
    return img


def capture_direction():
    img = np.array(ImageGrab.grab(bbox=direction_bbox))
    return img


# save format: date_key (key is one of w,a,s,d)
# for example, 2020-11-17-20:50:34_w
def save_data(data_img, control):
    # if control == 'w' or control == 'd': continue
    # save captured images in 'Autopilot/dataset/imgs/(file names)''
    target_directory = 'dataset/imgs'
    file_path = datetime.datetime.utcnow().strftime("%y-%m-%d:%H:%M:%S") + "_" + control + '.jpg'  # numpy array
    cv2.imwrite(file_path, data_img)


def main():
    while True:
        frontview = capture_frontview()
        map = capture_map()
        direction = capture_direction()
        keyinput = capture_key()
        cv2.imshow("Frontview", cv2.cvtColor(frontview, cv2.COLOR_BGR2RGB))
        cv2.imshow("Map", cv2.cvtColor(map, cv2.COLOR_BGR2RGB))
        cv2.imshow("Direction", cv2.cvtColor(direction, cv2.COLOR_BGR2RGB))
        print(keyinput)

        key = cv2.waitKey(1) & 0xFF
        # save_data(img, chr(key)) # lower case alphabet
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        # img = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        # cv2.imshow("Frame", original_img)
        # key = cv2.waitKey(1) & 0xFF
        # save_data(img, chr(key)) # lower case alphabet
        # if key == ord("q"):
        #     cv2.destroyAllWindows()
        #     break

main()
