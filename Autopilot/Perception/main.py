import os
import cv2
import time
import torch
import numpy as np
from PIL import Image, ImageDraw

# User-defined functions and classes
from Autopilot.Perception.ImageGrab import FrameCapture
import Autopilot.Perception.ImageProcess as ImageProcess
from Autopilot.Perception.ObjectDetection import YOLOv3_net
from Autopilot.Perception.Ultra_Fast_Lane_Detection import LaneDetection

# To avoid OpenMP related error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":

    # Load YOLOv5 from Pytorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().eval()  # yolov5s.pt
    model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS

    network = YOLOv3_net()
    img_w, img_h = 800, 600
    sct = FrameCapture((img_w, img_h), True, 2)

    while True:
        start_time = time.time()

        frame = sct.record_screen(ImageProcess.process_default)

        with torch.no_grad():
            prediction = model([frame], size=800)

        for i, (img, pred) in enumerate(zip([frame], prediction)):
            str = 'Image %g/%g: %gx%g ' % (i + 1, len([frame]), *img.shape[:2])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += '%g %ss, ' % (n, model.names[int(c)])  # add to string
                for *box, conf, cls in pred:  # xyxy, confidence, class
                    label = model.names[int(cls)] if hasattr(model, 'names') else 'class_%g' % cls
                    # str += '%s %.2f, ' % (label, conf)  # label
                    ImageDraw.Draw(img).rectangle(box, width=3)  # plot
                frame = np.array(img)

        # frame = LaneDetection.detect_lane(frame)
        fps_txt = 'FPS: %.1f' % (1. / (time.time() - start_time))
        print(fps_txt)
        cv2.imshow('Frame', frame)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    '''
    # YOLOv5 - Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().eval()  # yolov5s.pt
    model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS

    sct = FrameCapture((800, 600), True, 2)

    while True:
        start_time = time.time()
        frame = sct.record_screen(ImageProcess.process_default)

        with torch.no_grad():
            prediction = model(frame, size=900)  # includes NMS

        img = frame

        if prediction is not None:
            for c in prediction[:, -1].unique():
                n = (prediction[:, -1] == c).sum()  # detections per class
                str += '%g %ss, ' % (n, model.names[int(c)])  # add to string
            for *box, conf, cls in prediction:  # xyxy, confidence, class
                label = model.names[int(cls)] if hasattr(model, 'names') else 'class_%g' % cls
                str += '%s %.2f, ' % (label, conf)  # label
                ImageDraw.Draw(img).rectangle(box, width=3)  # plot

        fps_txt = 'FPS: %.1f' % (1. / (time.time() - start_time))
        print(fps_txt)
        cv2.imshow('Does it work..?', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        '''
