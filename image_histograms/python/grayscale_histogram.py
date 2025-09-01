# USAGE
# python grayscale_histogram.py --image beach.png

# import the necessary packages
from matplotlib import pyplot as plt
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# compute a grayscale histogram
# parameters: @image, @channels (here is just 0 channel), @mask (to compute only the mask region)...
# @histSize (number of bins), @ranges (in here the 256 is exclusive)
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

'''
hist will have along the x-axis the bins 0-256, and the y-axis tells how many these pixels intensities occur
'''
plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

# plot histogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixel values")
plt.plot(hist)
plt.xlim([0,256])
# above, if have 256x256 image. then all those pixels will get counted

# normalize - to show the frequency (percentage) instead of counting
hist /= image.sum()

plt.figure()
plt.title("Normalized Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixel values")
plt.plot(hist)
plt.xlim([0,256])

plt.show()
