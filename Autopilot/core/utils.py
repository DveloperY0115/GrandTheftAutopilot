import cv2 as cv2
import numpy as np
import mss
import time

# utility for capturing images


class FrameCapture:
    """
    FrameCapture class for capturing and doing elementary processing on game screen

    Takes target resolution, index of target monitor as input to generate an
    instance of FrameCapture class

    :param resolution: Resolution of captured frame expressed in (Width, Height)
    :param is_multi_monitor: Whether the system has two or more monitors
    :param target_monitor_idx: Index of monitor that game screen will be displayed

    :return an instance of FrameCapture class
    """

    def __init__(self, resolution=(800, 600), is_multi_monitor=False, target_monitor_idx=0):
        self.capturer = mss.mss()
        monitor_number = target_monitor_idx

        mon = self.capturer.monitors[monitor_number]
        # Using multiple monitors
        if is_multi_monitor:
            mon = self.capturer.monitors[monitor_number]

        self.monitor = {
            "top": mon["top"] + 40,  # 100px from the top
            "left": mon["left"],  # 100px from the left
            "width": resolution[0],
            "height": resolution[1],
            "mon": monitor_number,
        }

    def record_screen(self):
        """
        Captures a single frame

        :return frame: Numpy array holding pixel data of captured frame
        """
        start_time = time.time()

        frame = np.array(self.capturer.grab(self.monitor))
        frame = np.flip(frame[:, :, :3], 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print("fps: {}".format(1 / (time.time() - start_time)))
        return frame


class FrameProcessor:
    """
    FrameProcessor class for modifying obtained images from FrameCapture
    """
    def set_ROI(frame, vertices):
        """
        Set the region of interest in the given frame designated as set of vertices.
        :param frame: a frame to set ROI on
        :param vertices: Vertex points of ROI
        :return: Masked frame
        """
        # Create a mask of the size of input img
        mask = np.zeros_like(frame)

        # Fill the mask with designated vertices
        cv2.fillPoly(mask, vertices, 255)

        # Mask the img
        masked = cv2.bitwise_and(frame, mask)
        return masked

    @staticmethod
    def process_default(frame):
        """
        Default method for frame processing
        Do nothing
        :param frame: a frame to do processing on
        :return: same as input frame
        """
        original_frame = frame
        return original_frame

    @staticmethod
    def crop(image):
        """
        Crop the image (removing the sky at the top and the car front at the bottom)
        """
        return image[280:-130, :, :]

    @staticmethod
    def grayscale(img):
        """
        Applies the Grayscale transform
        This will return an image with only one color channel
        """
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def canny(img, low_threshold=100, high_threshold=300):
        """
        Applies the Canny transform
        """
        return cv2.Canny(img, low_threshold, high_threshold)

    @staticmethod
    def gaussian_blur(img, kernel_size):
        """
        Applies a Gaussian Noise kernel
        """
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=30, sigmaY=30)

    @staticmethod
    def process_frame(frame):
        """
        Processes the given frame using a set of elementary processing algorithms
        - RGB to Grayscale
        - Edge detection
        - Gaussian blur
        :param frame: a frame to do processing on
        :return: processed frame
        """

        # convert to gray
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # edge detection
        processed_frame = cv2.Canny(processed_frame, threshold1=200, threshold2=300)

        # apply Gaussian blur for clear line detection
        processed_frame = cv2.GaussianBlur(processed_frame, (5, 5), 0)

        mask_vertices = np.array([[10, 550], [10, 300], [300, 200],
                                  [500, 200], [800, 300], [800, 550]])
        processed_frame = FrameProcessor.set_ROI(processed_frame, [mask_vertices])

        return processed_frame
