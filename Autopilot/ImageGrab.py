import numpy as np
import cv2
import glob
import mss
import mss.tools
from PIL import Image
import time

def process_frame(frame):
    original_frame = frame

    # convert to gray
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # edge detection
    processed_frame = cv2.Canny(processed_frame, threshold1=200, threshold2=300)

    return processed_frame

def record_screen(resolution = (800, 600), is_multi_monitor = False, target_monitor_idx = 1,):
    # Record resolution : 800 x 600
    capturer = mss.mss()
    monitor_number = target_monitor_idx

    mon = capturer.monitors[monitor_number]
    # Using multiple monitors
    if is_multi_monitor:
        mon = capturer.monitors[monitor_number]

    monitor = {
        # Captures the top left-most 800 x 600 of the monitor region
        "top": mon["top"] + 40,  # 100px from the top
        "left": mon["left"],  # 100px from the left
        "width": resolution[0],
        "height": resolution[1],
        "mon": monitor_number,
    }

    # Initialize timer to capture the interval between each process
    previous_time = 0
    while True:
        frame = np.array(capturer.grab(monitor))
        cv2.imshow('Captured Frame', process_frame(frame))

        if cv2.waitKey( 1 ) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        fps_txt = 'FPS: %.1f' % ( 1./(time.time() - previous_time))
        previous_time = time.time()
        print(fps_txt)
