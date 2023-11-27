import cv2
import numpy as np


def cv2_read(img_path):
    full_img = cv2.imread(img_path)
    # Convert the image from BGR (cv2 default loading style) to RGB
    full_img = np.array(full_img[..., ::-1])
    return full_img
