# USAGE
# python predict.py --model output/model.pth

# set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)

# import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset # to get a subset of the data loader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to the trained PyTorch model")
args = vars(ap.parse_args())

# set the device we will be using to test the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

print("[INFO] loading KMNIST dataset...")
testData = KMNIST(root="data", train=False, download=True, transform=ToTensor())
idx = np.random.choice(range(0, len(testData)), size=(10,))
testData = Subset(testData, idx) # generate a subset from 10 random indices obtained previously

testDataLoader = DataLoader(testData, batch_size=1) # loop one by one over the data

with torch.no_grad():
    model = torch.load(args["model"], weights_only=False).to(device)
    model.eval()

    for (image, label) in testDataLoader:
        # grab original image and ground truth label
        origImage = image.numpy().squeeze(axis=(0,1))
        gtLabel = testData.dataset.classes[label.numpy()[0]]

        image = image.to(device)
        pred = model(image)

        # find the class label
        idx = pred.argmax(axis=1).cpu().numpy()[0] # find label with largest probability
        predLabel = testData.dataset.classes[idx] # get label from dataset

        # convert binary image to rgb (so is more easily drawn and see)
        origImage = np.dstack([origImage] * 3)
        origImage = imutils.resize(origImage, width=128) # resize image to 128 width

        # draw the predicted class label on it
        color = (0, 255, 0) if gtLabel == predLabel else (0, 0, 255)
        cv2.putText(origImage, gtLabel, (2, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
        # display the result in terminal and show the input image
        print("[INFO] ground truth label: {}, predicted label: {}".format(
            gtLabel, predLabel))
        cv2.imshow("image", origImage)
        cv2.waitKey(0)