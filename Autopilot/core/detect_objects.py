import os
import sys
project_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(project_dir)


# import modules
import time
from absl import logging
import cv2
import numpy as np
import tensorflow as tf

from Autopilot.library.yolov3_tf2.yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from Autopilot.library.yolov3_tf2.yolov3_tf2.dataset import transform_images
from Autopilot.library.yolov3_tf2.yolov3_tf2.utils import draw_outputs


class Detector:
    """
    A detector class for detecting & identifying objects in the given image.

    This classes uses 'YOLOv3' for real time inferencing.
    """
    def __init__(self,
                 classes='./library/yolov3_tf2/data/coco.names',
                 weights='./library/yolov3_tf2/checkpoints/yolov3.tf',
                 tiny=False, size=416, num_classes=80
                 ):
        """
        Initializes detector class using options provided by user
        :param classes: String. Path to the file which contains the information of detectable classes
        :param weights: String. Path to tensorflow weights file
        :param tiny: Boolean. If True, initialize detector with smaller version of YOLO
        :param size: Int. The size which the input image will be resized to
        :param num_classes: Number of classes the model aims to distinguish

        By default,

        """
        self.physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in self.physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)

        if tiny:
            self.model = YoloV3Tiny(classes=num_classes)
            logging.info('Using YoloV3Tiny')
        else:
            self.model = YoloV3(classes=num_classes)
            logging.info('Using YoloV3')

        self.model.load_weights(weights)     # load weights from specified path
        logging.info('weights loaded')

        self.size = size

        self.class_names = [c.strip() for c in open(classes).readlines()]    # load classes from specified path
        logging.info('classes loaded')

    def detect(self, input_img):
        if input_img is None:
            logging.warning("Empty Frame")

        img_raw = tf.convert_to_tensor(input_img)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, self.size)

        t1 = time.time()
        boxes, scores, classes, nums = self.model(img)

        t2 = time.time()
        logging.info('Inference time: {}'.format(t2 - t1))

        logging.info('detections:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(self.class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

        img = img_raw.numpy()
        img = draw_outputs(img, (boxes, scores, classes, nums), self.class_names)

        return img
