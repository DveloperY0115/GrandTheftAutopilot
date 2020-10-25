import numpy as np
import cv2
import mss
import mss.tools
import time
import ImageProcess


class FrameCapture:
    """
    FrameCapture class for capturing and doing elementary processing on game screen

    Takes target resolution, index of target monitor as input to generate an
    instance of FrameCapture class

    :arg
        resolution (tuple): Resolution of captured frame expressed in (Width, Height)
        is_multi_monitor (bool): Whether the system has two or more monitors
        target_monitor_idx (int): Index of monitor that game screen will be displayed

    :returns
        an instance of FrameCapture class
    """
    def __init__(self, resolution=(800, 600), is_multi_monitor=False,
                  target_monitor_idx=1):
        self.capturer = mss.mss()
        monitor_number = target_monitor_idx

        mon = self.capturer.monitors[monitor_number]
        # Using multiple monitors
        if is_multi_monitor:
            mon = self.capturer.monitors[monitor_number]

        self.monitor = {
            # Captures the top left-most 800 x 600 of the monitor region
            "top": mon["top"] + 40,  # 100px from the top
            "left": mon["left"],  # 100px from the left
            "width": resolution[0],
            "height": resolution[1],
            "mon": monitor_number,
        }

    def record_screen(self, processing_method=ImageProcess.process_default):
        """
        Captures a single frame

        :arg
            processing_method (function): Type of frame processing defined in ImageProcess.py

        :returns
            frame(numpy array): Numpy array holding pixel data of captured frame
        """

        # Initialize timer to capture the interval between each process
        previous_time = 0
        frame = np.array(self.capturer.grab(self.monitor))
        frame = np.flip(frame[:, :, :3], 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Captured Frame', processing_method(frame))

        # Elapsed time to capture a frame
        fps_txt = 'FPS: %.1f' % (1./(time.time() - previous_time))
        print(fps_txt)

        return frame


