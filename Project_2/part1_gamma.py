import numpy as np 
import cv2
from matplotlib import pyplot as plt

def gamma_correct(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    # First, pixel intensities must be scaled from the range [0, 255] to [0, 1.0]
    # Then do gamma correction
    # Scale back the image to range [0, 255]
    gamma_inv = 1.0 / gamma
    lookup_table = np.array([((it / 255.0) ** gamma_inv) * 255
        for it in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, lookup_table)

def main():
    cap = cv2.VideoCapture('Night Drive - 2689.mp4')
    gamma = 2

    while(cap.isOpened()):
        ret, original = cap.read()
            
        corrected = gamma_correct(original, gamma)
        corrected = cv2.GaussianBlur(corrected,(3,3),0)

        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)
        cv2.imshow('image', np.hstack([original, corrected]))
        cv2.waitKey(1)
main()
