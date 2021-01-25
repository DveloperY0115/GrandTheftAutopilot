import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
from DataLoader import DataLoader

if __name__ == "__main__":
    path_dir = '..\dataset'

    loader = DataLoader(path_to_datasets=path_dir, use_colab=False)

    X, y = loader.load_data("210125_04-18-29_data")

    test_img = X[11].reshape(594, 800, 3)
    for i in range(len(y)):
        print(y[i])
    while True:
        cv2.imshow("img", test_img)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
