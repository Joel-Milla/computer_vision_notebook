# USAGE
# python detect_apriltag.py --image images/example_01.png

# import the necessary packages
import apriltag
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image containing AprilTag")
args = vars(ap.parse_args())

# Load image and convert to grayscale, only preprocessing step needed
print("[INFO] preprocessing step")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detecting april tags detectors and then aprilTags in input image
print("[INFO] detecting April tags")
options = apriltag.DetectorOptions(families="tag36h11") # select the family of the aprilTags being detected
detector = apriltag.Detector(options)
results = detector.detect(gray)
print("[INFO] {} total Apriltags detected".format(len(results)))

# loop over results
for result in results:
    (ptA, ptB, ptC, ptD) = result.corners
    ptA = (int(ptA[0]), int(ptA[1])) # convert numpy integers to normal integers
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))

    cv2.line(image, ptA, ptB, (0,255,0), 2)
    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
    cv2.line(image, ptD, ptA, (0, 255, 0), 2)

    (cx, cy) = (int(result.center[0]), int(result.center[1])) # get center of aprilTag
    cv2.circle(image, (cx, cy), 5, (0,0,255), -1)

    # draw tag family on the image
    tagFamily = result.tag_family.decode("utf-8")
    cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 2)
    print("[INFO] tag family: {}".format(tagFamily))


# show output
cv2.imshow("Image", image)
cv2.waitKey()