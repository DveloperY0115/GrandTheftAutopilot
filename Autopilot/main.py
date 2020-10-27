import cv2
import time

# User-defined functions and classes
from ImageGrab import FrameCapture
import ImageProcess
from ObjectDetection import YOLOv3_net


if __name__ == "__main__":

    network = YOLOv3_net()
    sct = FrameCapture((800, 600), True, 1)

    while True:
        start_time = time.time()
        frame = sct.record_screen(ImageProcess.process_default)

        frame = network.detect_objects(frame)
        fps_txt = 'FPS: %.1f' % (1. / (time.time() - start_time))
        print(fps_txt)
        cv2.imshow('Does it work..?', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break