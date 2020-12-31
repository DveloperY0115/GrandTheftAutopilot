import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2 as cv2
from absl import app
from Autopilot.core.utils import FrameCapture, FrameProcessor
from Autopilot.core import detect


def main(_argv):

    # initialize capture/detection tools

    # TODO: Format of file path is different on Windows and Linux/macOS
    main_abs = os.path.abspath(os.path.dirname(__file__))
    class_abs = os.path.join(main_abs, "library\yolov3_tf2\data\coco.names")
    weight_abs = os.path.join(main_abs, "library\yolov3_tf2\checkpoints\yolo3.tf")

    print(class_abs)
    print(weight_abs)

    capture = FrameCapture(resolution=(800,600), is_multi_monitor=True, target_monitor_idx=0)

    detector = detect.Detector(
        classes=class_abs,
        weights=weight_abs,
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
