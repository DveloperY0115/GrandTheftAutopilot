import os
import cv2
import numpy as np
import datetime
# import Autopilot.Perception.ImageGrab as ImageGrab
from PIL import ImageGrab, Image
import win32api as wapi
import pandas as pd

# 1280 * 720


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class DataCollect:
    #### Global Variables ####
    frontview_bbox = (0, 40, 800, 640)
    mapview_bbox = (5, 520, 160, 630)
    direction_bbox = (15, 610, 25, 620)
    img_w, img_h = 800, 600

    keyList = ["\b"]
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789,.'£$/\\":
        keyList.append(char)

    def __init__(self):
        pass

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
        img = np.array(ImageGrab.grab(bbox=frontview_bbox))
        # Preprocess needed
        return img


    def capture_mapview():
        img = np.array(ImageGrab.grab(bbox=mapview_bbox))
        return img


    def capture_direction():
        img = np.array(ImageGrab.grab(bbox=direction_bbox))
        return img


    # save format: date_key (key is one of w,a,s,d)
    # for example, 2020-11-17-20:50:34_w
    def save_data(frontview_img, mapview_img, direction_img, control):
        # if control == 'w' or control == 'd': continue
        # save captured images in 'Autopilot/dataset/imgs/(file names)''
        target_directory = './dataset/imgs/'
        frontview_filename = datetime.datetime.utcnow().strftime(
            "%y-%m-%d_%H_%M_%S") + "_" + "frontview" + '.jpg'  # numpy array
        mapview_filename = datetime.datetime.utcnow().strftime(
            "%y-%m-%d_%H_%M_%S") + "_" + "mapview" + '.jpg'  # numpy array
        direction_filename = datetime.datetime.utcnow().strftime(
            "%y-%m-%d_%H_%M_%S") + "_" + "direction" + '.jpg'  # numpy array
        # frontview_img = Image.fromarray(frontview_img)
        # frontview_img.save(target_directory + frontview_filename)
        print('directory: ', target_directory + frontview_filename)
        cv2.imwrite(target_directory + frontview_filename, frontview_img)
        cv2.imwrite(target_directory + mapview_filename, mapview_img)
        cv2.imwrite(target_directory + direction_filename, direction_img)

        temp_dict = {'frontview': frontview_filename, 'mapview': mapview_filename,
                     'direction': direction_filename, 'control': control, 'speed': 0}
        return temp_dict


if __name__ == '__main__':    # 프로그램의 시작점일 때만 아래 코드 실행
    index = 0
    data_dict = {}
    while True:
        frontview = capture_frontview()
        mapview = capture_mapview()
        direction = capture_direction()
        keyinput = capture_key()
        frontview = cv2.cvtColor(frontview, cv2.COLOR_BGR2RGB)
        mapview = cv2.cvtColor(mapview, cv2.COLOR_BGR2RGB)
        direction = cv2.cvtColor(direction, cv2.COLOR_BGR2RGB)
        cv2.imshow("Frontview", frontview)
        cv2.imshow("mapview", mapview)
        cv2.imshow("Direction", direction)
        print(keyinput)
        data_log = save_data(frontview, mapview, direction, keyinput)
        print(data_log)
        index += 1
        data_dict[index] = data_log

        key = cv2.waitKey(1) & 0xFF
        # save_data(img, chr(key)) # lower case alphabet
        if key == ord("q"):
            cv2.destroyAllWindows()
            dataset = pd.DataFrame.from_dict(data_dict, orient='index')
            dataset_name = datetime.datetime.utcnow().strftime("%y-%m-%d_%H_%M_%S")+"_dataset.csv"
            dataset.to_csv(dataset_name)
            break
