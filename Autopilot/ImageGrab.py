import numpy as np
from PIL import ImageGrab
import cv2


def record_screen():
    # Record resolution : 800 x 600
    while True:
        screen_pil = ImageGrab.grab(bbox=(0, 40, 800, 640))
        screen_numpy = np.array(screen_pil.getdata(), dtype='uint8')\
            .reshape((screen_pil.size[1], screen_pil.size[0], 3))
        cv2.imshow('window', screen_numpy)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
