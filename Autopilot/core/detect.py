import os, sys
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, 'library/yolov3-tf2/yolov3_tf2'))

# import modules
import time
import cv2
import numpy as np
import tensorflow as tf



print(os.path.join(project_dir, 'library/yolov3-tf2'))