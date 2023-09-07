import tensorflow as tf
import cv2
import imutils
from imutils import paths
import numpy as np
import random

from ContrastBrightness import ContrassBrightness
from Clusterer import Clusterer

import os
from skimage.morphology import skeletonize
from sklearn.metrics import classification_report

CONTRASTER = ContrassBrightness()
CLUSTERER = Clusterer()

def white_percent(img):
    # calculated percent of white pixels in the grayscale image
    w,h = img.shape
    total_pixels = w * h

    # counts the numbers of white pixels
    white_pixels = 0

    for r in img:
        for c in r:
            if c == 255:
                white_pixels += 1
    return white_pixels/total_pixels

def fix_image(img):
    # inversion
    img = cv2.bitwise_not(img)

    # thresholding
    image_bw = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)[1]

    # making mask of a circle
    black = np.zeros((250, 250))# our images are 250 * 250 when we are transforming them
    # circle center is (125, 125), radius is 110, color is white
    # we divide by 255 to get image of 1s and 0s. Where a 0s is seen, our image will become black
    # make our white outside black and keep the white digit as it is inside the circle
    circle_mask = cv2.circle(black, (125, 125), 110, (255, 255, 255), -1) / 255.0

    # applying mask to make everything outside the circle black
    edited_image = image_bw * (circle_mask.astype(image_bw.dtype))
    return edited_image


#used later as inputs to the neural network model
proccessed_images = []
image_labels = []

#prepare image paths and randomizes them
image_paths = list(paths.list_images("charts/ordered"))
random.shuffle(image_paths)

for imagePath in image_paths:
    image = cv2.imread(imagePath)

    #resize
    image = imutils.resize(image, height=250)

    #contrast (no change in brightness)
    image = CONTRASTER.apply(image, 0, 60)

    #blurring
    image = cv2.medianBlur(image, 15)
    image = cv2.GaussianBlur(image, (3,3), cv2.BORDER_DEFAULT)

    #clustering
    image = CLUSTERER.apply(image, 5)

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = 0

    percent_white = white_percent(cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1])
    while (not (percent_white > 0.10 and percent_white < 0.28)) and threshold <= 255:
        threshold += 10
        percent_white = white_percent(cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1])

    #show the image
    cv2.imshow("Gray", gray)
    cv2.waitKey()
    cv2.DestroyAllWindows()







