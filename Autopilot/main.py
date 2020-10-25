import numpy as np
import cv2
import mss
import mss.tools
import time
import pyautogui

# User-defined functions and classes
from DirectKeys import PressKey, W, A, S, D
from ImageGrab import FrameCapture
import ImageProcess
from ObjectDetection import YOLO_net


if __name__ == "__main__":

    network = YOLO_net()
    sct = FrameCapture((800, 600), True, 1)

    while True:
        frame = sct.record_screen(ImageProcess.process_default)

        frame = network.detect_objects(frame)
        cv2.imshow('Does it work..?', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break