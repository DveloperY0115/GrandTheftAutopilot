import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import cv2
import numpy as np
from absl import app

from Autopilot.core import img_process
from Autopilot.core import detect
# from darkflow.net.build import TFNet


def main(_argv):

    # tfnet = TFNet(options)

    # initialize capture/detection tools

    # TODO: Format of file path is different on Windows and Linux/macOS
    main_abs = os.path.abspath(os.path.dirname(__file__))
    class_abs = os.path.join(main_abs, "library\yolov3_tf2\data\coco.names")
    weight_abs = os.path.join(main_abs, "library\yolov3_tf2\checkpoints\yolo3.tf")

    detector = detect.Detector(
        classes=class_abs,
        weights=weight_abs,
        tiny=False, size=416, num_classes=80
    )

    processor = img_process.Image_Processor()

    while True:
        t0 = time.time()
        screen = processor.grab_screen().astype(np.uint8)
        screen = detector.detect(screen)
        # result = tfnet.return_predict(screen)

        """
        frame = capture.record_screen()
        frame = detector.detect(frame)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        """

        ellapsed = time.time() - t0
        print(ellapsed, "sec")
        cv2.imshow("Capture", screen)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
