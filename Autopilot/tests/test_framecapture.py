if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2 as cv2

import core.utils as utils

myCapture = utils.FrameCapture()

while True:
    frame = myCapture.record_screen()
    frame = utils.FrameProcessor.process_frame(frame)
    cv2.imshow("test", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

