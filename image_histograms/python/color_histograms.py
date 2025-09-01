# USAGE
# python color_histograms.py --image beach.png

# import the necessary packages
from matplotlib import pyplot as plt
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
plt.figure()
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

chans = cv2.split(image)
colors = ("b", "g", "r")

plt.figure()
plt.title("Flattened color histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")

# loop over image channels
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

# create a new figure
fig = plt.figure()
ax = fig.add_subplot(131)
# channels b and g. compute the channles of 0 and 1
# hit size is making 32x32 matrix. and the ranges are the ranges of each individual channel (0-255
hist = cv2.calcHist([chans[1], chans[0]], [0,1],
                    None, [32,32],
                    [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2d color histogram for G and B")
plt.colorbar(p)

# for green and red
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0,1],
                    None, [32,32],
                    [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2d color histogram for G and B")
plt.colorbar(p)

# for blue and red
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0,1],
                    None, [32,32],
                    [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2d color histogram for G and B")
plt.colorbar(p)

plt.show()


