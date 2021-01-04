import win32gui
import win32ui

import cv2
import numpy as np
import win32con


class Image_Processor:

    def __init__(self):
        """
        Constructor.

        Create two KNN models performing inference on speedometer, direction indicator
        """
        def initKNN(self, data, labels, shape):
            knn = cv2.ml.KNearest_create()
            train_data = np.load(data).reshape(-1, shape).astype(np.float32)
            train_labels = np.load(labels)
            knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
            return knn

        # self.knnDigits = initKNN("", "", 40)
        # self.knnArrows = initKNN("", "", 90)

    def grab_screen(self, winName: str = "Grand Theft Auto V"):
        """
        grab_screen

        Grab game screen in the window named 'winName'
        :param winName: Name of the game window to be captured
        :return: Numpy array
        """
        desktop = win32gui.GetDesktopWindow()

        # get area by a window name
        gtawin = win32gui.FindWindow(None, winName)
        # get the bounding box of the window
        x_left, y_up, x_right, y_down = win32gui.GetWindowRect(gtawin)
        # cut window boarders
        """
        Note: Windows coordinate system
        (origin)------------> x_right, y_up
        |
        |
        v   x_left, y_down
        """
        y_up += 32
        x_left += 3
        y_down -= 4
        x_right -= 4
        width = x_right - x_left + 1
        height = y_down - y_up + 1

        # the device context(DC) for the entire window (title bar, menus, scroll bars, etc.)
        hwindc = win32gui.GetWindowDC(desktop)
        # Create a DC object from an integer handle
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        # Create a memory device context that is compatible with the source DC
        memdc = srcdc.CreateCompatibleDC()
        # Create a bitmap object
        bmp = win32ui.CreateBitmap()
        # Create a bitmap compatible with the specified device context
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        # Select an object into the device context.
        memdc.SelectObject(bmp)
        # Copy a bitmap from the source device context to this device context
        # parameters: destPos, size, dc, srcPos, rop(the raster operation))
        memdc.BitBlt((0, 0), (width, height), srcdc, (x_left, y_up), win32con.SRCCOPY)

        # the bitmap bits
        signedIntsArray = bmp.GetBitmapBits(True)
        # form a 1-D array initialized from text data in a string.
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        # Delete all resources associated with the device context
        srcdc.DeleteDC()
        memdc.DeleteDC()
        # Releases the device context
        win32gui.ReleaseDC(desktop, hwindc)
        # Delete the bitmap and freeing all system resources associated with the object.
        # After the object is deleted, the specified handle is no longer valid.
        win32gui.DeleteObject(bmp.GetHandle())

        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    def predict(self, img, knn):
        ret, result, neighbours, dist = knn.findNearest(img, k=1)
        return result

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -5)
        return thr

    def convert_speed(self, num1, num2, num3):
        hundreds = 1
        tens = 1
        speed = 0

        if num3[0][0] != 10:
            hundreds = 10
            tens = 10
            speed += int(num3[0][0])
        if num2[0][0] != 10:
            speed += tens * int(num2[0][0])
            hundreds = tens * 10
        if num1[0][0] != 10:
            speed += hundreds * int(num1[0][0])

        return speed

    def img_process(self, winName: str = "Grand Theft Auto V"):
        screen = self.grab_screen(winName)

        numbers = self.preprocess(screen[567:575, 683:702, :])

        # three fields for numbers
        num1 = self.predict(numbers[:, :5].reshape(-1, 40).astype(np.float32), self.knnDigits)
        num2 = self.predict(numbers[:, 7:12].reshape(-1, 40).astype(np.float32), self.knnDigits)
        num3 = self.predict(numbers[:, -5:].reshape(-1, 40).astype(np.float32), self.knnDigits)

        direct = self.preprocess(screen[561:570, 18:28, :]).reshape(-1, 90).astype(np.float32)
        direct = int(self.predict(direct, self.knnArrows)[0][0])

        speed = self.convert_speed(num1, num2, num3)
        resized = cv2.resize(screen, (320, 240))

        return screen, resized, speed, direct




