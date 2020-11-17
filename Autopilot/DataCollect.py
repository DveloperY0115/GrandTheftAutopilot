import os
import cv2
import numpy as np
import datetime
# import Autopilot.Perception.ImageGrab as ImageGrab
from PIL import ImageGrab

cwd = os.getcwd()

img_w, img_h = 800,600

# save format: date_key (key is one of w,a,s,d)
# for example, 2020-11-17-20:50:34_w
def save_data(data_img, control):
    # if control == 'w' or control == 'd': continue
    file_path = str(cwd) + '/' + datetime.datetime.utcnow().strftime("%y-%m-%d:%H:%M:%S") + '.jpg' # numpy array
    cv2.imwrite(file_path, data_img)


def main():
    while True:
        img = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        cv2.imshow("Frame", original_img)
        key = cv2.waitKey(1) & 0xFF
        save_data(img, chr(key)) # lower case alphabet
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
