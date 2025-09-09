# USAGE
# python train.py --model output/model.pth --plot output/plot.png

# set the matplotlib backend so figures can be saved in the background
# this is when saving plots without rendering them. just to save it and don't show it....
# tells it to use non-interactive backends
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.lenet import LeNet # what we implemented
from sklearn.metrics import classification_report # to generate report
from torch.utils.data import random_split # able to create train/test split function
from torch.utils.data import DataLoader # makes it easy for data processing pipelines
from torchvision.transforms import ToTensor # takes data and transform it to pytorch tensor to pass to network
from torchvision.datasets import KMNIST # dataset
from torch.optim import Adam # optimzer for training
from torch import nn # neural netowrk functinality
import matplotlib.pyplot as plt # plotting
import numpy as np
import argparse
import torch
import time # to time how long it takes

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# define training hyperparameters
INIT_LR = 1e-3 # standard learning rate when using adam optimizer
BATCH_SIZE = 64
EPOCHS = 10

#define train test splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Get the data from the folder 'data'. If doesn't exist, the KMNIST will download it and ...
# save it in a folder with that name in your directory. PyTorch will download and cache it
print("[INFO] loading KMNIST dataset...")
trainData = KMNIST(root="data", train=True, download=True, transform=ToTensor())
testData = KMNIST(root="data", train=False, download=True, transform=ToTensor())

print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
# split, with lengths of train/val samples. and then use seed for reproducibility
(trainData, valData) = random_split(trainData, [numTrainSamples, numValSamples]
                                    , generator=torch.Generator().manual_seed(42))

# initialize train, validation, and test data loaders
'''
The data loader is to loop in that instead of this: 
for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE)
'''
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps for epoch
trainSteps = len(trainDataLoader.dataset) # batch size
valSteps = len(valDataLoader.dataset) # batch size

print("[INFO] initializing the LeNet Model")
model = LeNet(
    numChannels=1,
    classes=len(trainData.dataset.classes)).to(device)

# initialize optimzer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

# initialize a dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()
for e in range(0, EPOCHS):
    # set model to training mode
    model.train()

    # bookkeeping variables
    totalTrainLoss = 0
    totalValLoss = 0

    # number of correct predictions in training and validation set
    trainCorrect = 0
    valCorrect = 0

    for (x, y) in trainDataLoader:
        (x, y) = (x.to(device), y.to(device))

        # perform forward pass and calculate training loss
        pred = model(x)
        loss = lossFn(pred, y)

        # update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    # now that finished one epoch, lets see how model is performing in the validation set
    with torch.no_grad():
        model.eval() # need to set to eval, to evaluate and make predictions

        for (x, y) in valDataLoader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = lossFn(pred, y)

            totalValLoss += loss
            '''
            pred = tensor([[0.1, 0.8, 0.1],    # Sample 1: probabilities for 3 classes
               [0.9, 0.05, 0.05],  # Sample 2: probabilities for 3 classes  
               [0.2, 0.3, 0.5]])   # Sample 3: probabilities for 3 classes
            argmax(1) returns maximum index of maximum value a long axis=1 (columns)
            thus: (pred.argmax(1) == y) returns an array of True,False of checking
            whether index predictions were correct. Then convert that to float, so...
            the True/False can be summed and obtained total correct predictions.
            then, item() converts to python scalar, as previously it was a torch tensor
            '''
            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    # calculate average training/validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # calculate training and validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)

    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)

    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
        avgValLoss, valCorrect))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

print("[INFO] evaluating network")
totalTestLoss = 0
testCorrect = 0


with torch.no_grad():
    model.eval()  # need to set to eval, to evaluate and make predictions
    preds = []

    for (x, y) in testDataLoader:
        x = x.to(device)
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

# generate classification report
print(classification_report(testData.targets.cpu().numpy(),
                            np.array(preds), target_names=testData.classes))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
torch.save(model, args["model"]) # save the model with the name that was an input argument