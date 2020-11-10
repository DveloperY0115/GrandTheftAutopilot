import cv2
import time
import torch
import numpy as np
from PIL import Image, ImageDraw

# User-defined functions and classes
from ImageGrab import FrameCapture
import ImageProcess
from ObjectDetection import YOLOv3_net
from Ultra_Fast_Lane_Detection import LaneDetection



if __name__ == "__main__":
    network = YOLOv3_net()
    img_w, img_h = 800, 600
    sct = FrameCapture((img_w, img_h), True, 1)

    while True:
        frame = sct.record_screen(ImageProcess.process_default)
        out = network.detect_objects(frame)
        frame = network.draw_boxes(frame, out[0], out[1], out[2])
        frame = LaneDetection.detect_lane(frame)
        print(type(frame))
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
