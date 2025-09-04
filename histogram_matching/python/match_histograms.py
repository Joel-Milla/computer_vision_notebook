# USAGE
# python match_histograms.py --source empire_state_cloudy.png --reference empire_state_sunset.png
# import the necessary packages
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,
	help="path to the input source image")
ap.add_argument("-r", "--reference", required=True,
	help="path to the input reference image")
args = vars(ap.parse_args())

# load the source and reference images
print("[INFO] loading source and reference images...")
src = cv2.imread(args["source"])
ref = cv2.imread(args["reference"]) # the one that has the histogram we want to transfer

print("[INFO] Performing histogram matching")
multi = (len(src) - 1) if src.shape[-1] > 1 else None # check if it has multiple channels
matched = exposure.match_histograms(src, ref, channel_axis=2)
print(src.shape)
cv2.imshow("Source", src)
cv2.imshow("Reference", ref)
cv2.imshow("Matched", matched)
cv2.waitKey()

(fig, axs) = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

# loop over the three images
for (i, image) in enumerate((src, ref, matched)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for (j, color) in enumerate(("red", "green", "blue")):
        (hist, bins) = exposure.histogram(image[..., j],
                                          source_range="dtype")
        axs[j, i].plot(bins, hist / hist.max())

        # compute cumulative distribution of current channel
        (cdf, bins) = exposure.cumulative_distribution(image[..., j])
        axs[j, i].plot(bins, cdf)
        # set y-axis to be the name of the current channel
        axs[j, 0].set_ylabel(color)
# set axes titles
axs[0, 0].set_title("Source")
axs[0, 1].set_title("Reference")
axs[0, 2].set_title("Matched")

plt.tight_layout()
plt.show()
