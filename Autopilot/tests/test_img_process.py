import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2 as cv2
from Autopilot.core import img_process

while True:
    processor = img_process.Image_Processor()

    frame = processor.grab_screen()

    cv2.imshow("searching for regions", frame)
    
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

