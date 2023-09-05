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

    #show the image
    cv2.imshow("Gray", gray)
    cv2.waitKey()
    cv2.DestroyAllWindows()