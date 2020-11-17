import os
import sys

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from Autopilot.Perception.ImageGrab import FrameCapture
import Autopilot.Perception.ImageProcess as ImageProcess
from Autopilot.Perception.ObjectDetection import YOLOv3_net
from Autopilot.Perception.Ultra_Fast_Lane_Detection import LaneDetection

from Autopilot.Perception.YOLOv5.models.experimental import attempt_load
from Autopilot.Perception.YOLOv5.utils.datasets import LoadStreams, LoadImages
from Autopilot.Perception.YOLOv5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from Autopilot.Perception.YOLOv5.utils.plots import plot_one_box
from Autopilot.Perception.YOLOv5.utils.torch_utils import select_device, load_classifier, time_synchronized

sys.path.insert(0, './YOLOv5')

# To avoid OpenMP related error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class YOLOv5_net:
    def __init__(self, weights_dir='yolov5s.pt', imgsz=800):
        """
        Initialize the YOLOv5 model with given weights and input size
        :param weights_dir: File name of pytorch weight
        :param imgsz: Size of input image
        """

        # Initialize
        set_logging()
        self.device = select_device()
        self.isHalf = self.device.type != 'cpu'

        # Load Model
        self.model = attempt_load(weights_dir, map_location=self.device)
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())

        if self.isHalf:
            self.model.half()

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)
            self.modelc.load_state_dict(torch.load('YOLOv5/weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def detect(self, frame):
        """
        Given the frame, detect objects in the scene and classify them
        :param frame: Numpy array form of image
        :return: Tensor containing bounding box predictions and modified / original image
        """
        t0 = time.time()

        # Test
        annotated = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(annotated.half() if self.isHalf else annotated) if self.device.type != 'cpu' else None

        annotated = torch.from_numpy(frame).permute(2, 0, 1).to(self.device)
        annotated = annotated.half() if self.isHalf else annotated.float()
        annotated /= 255.0
        if annotated.ndimension() == 3:
            annotated = annotated.unsqueeze(0)

        # Run inference
        t1 = time_synchronized()
        pred = self.model(annotated)[0]

        # Apply NMS
        # ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']
        target_class = [0, 1, 2, 3, 5, 7, 9]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=target_class)
        t2 = time_synchronized()

        # Apply classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, annotated, frame)

        print('Elapsed time for inference %.1f' % (t2-t1))
        return pred, annotated, frame

    def plot_boxes(self, pred, annotated, frame):
        """
        TODO: use of 'annotated' seems a bit redundant
        Draw bounding boxes and their labels (with confidence) on the given image
        :param pred: Tensor containing bounding box predictions
        :param annotated: An annotated image
        :param frame: An original image
        :return: An image containing bounding boxes and their labels
        """
        t0 = time.time()
        if pred[0] is None:
            # Nothing is detected
            # DO NOTHING
            return frame

        # Process detections
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(annotated.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, frame, label=label, color=self.colors[int(cls)], line_thickness=3)
        t1 = time.time()

        print('Elapsed time for plotting %.1f' % (t1 - t0))
        return frame

if __name__ == '__main__':

    net = YOLOv5_net()
    img_w, img_h = 640, 640
    sct = FrameCapture((img_w, img_h), is_multi_monitor=True, target_monitor_idx=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    while True:
        t0 = time.time()
        frame = sct.record_screen(ImageProcess.process_default)

        with torch.no_grad():
            result = net.detect(frame)

        cv2.imshow('Frame', net.plot_boxes(result[0], result[1], result[2]))

        # frame = LaneDetection.detect_lane(frame)
        fps_txt = 'FPS: %.1f' % (1. / (time.time() - t0))
        print(fps_txt)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
