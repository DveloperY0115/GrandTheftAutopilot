import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2 as cv2
from absl import app
from Autopilot.core.utils import FrameCapture, FrameProcessor
from Autopilot.core import detect


def main(_argv):

    # initialize capture/detection tools

    capture = FrameCapture()

    detector = detect.Detector(
        classes='./library/yolov3_tf2/data/coco.names',
        weights='./library/yolov3_tf2/checkpoints/yolov3.tf',
        tiny=False, size=416, num_classes=80
    )

    while True:
        frame = capture.record_screen()
        frame = detector.detect(frame)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
