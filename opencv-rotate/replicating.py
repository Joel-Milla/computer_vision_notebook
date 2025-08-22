#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 20:35:14 2025

@author: joel
"""

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="opencv_logo.png",
	help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("og", image)
cv2.waitKey(0)

(h, w) = image.shape[:2]
(cx, cy) = (w // 2, h // 2)

M = cv2.getRotationMatrix2D((cx, cy), 180, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated 180 ", rotated)
cv2.waitKey(0)
