from pyimagesearch import mlp # model defined inside pyimagesearch
from torch.optim import SGD # stochastic gradient descent optimizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs # to create sample dataset
import torch.nn as nn
import torch

def next_batch(inputs, targets, batchSize):
    for i in range(0, inputs.shape[0], batchSize):
        '''
        - this gives (yields, returns) data back to the callback function
        - here its giving the first 'batchSize's inputs and its targets, then...
        the next 'batchSize's inputs and targets, etc. Until reach the end
        '''
        yield (inputs[i:i + batchSize], targets[i:i + batchSize])


BATCH_SIZE=64
EPOCHS = 10
LR = 1e-2

# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}".format(DEVICE))

print("[INFO] preparing data...")
# 1,000 samples, with each sample having 4 features, 3 centers...
# and std of spread of data points inside the centers, and reproducibility random number
(X, y) = make_blobs(n_samples=1000, n_features=4, centers=3, cluster_std=2.5,
                    random_state=95)

# create train, test, split
trainX, textX, trainY, testY = train_test_split(X, y,
                                                test_size=0.15, random_state=95)

trainX = torch.from_numpy(trainX).float() # convert np arrays to torch tensors (torch arrays)
testX = torch.from_numpy(textX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()

# initialize our model and display its architecture
mlp = mlp.get_training_model().to(DEVICE) # this takes model and moves it to the device use for training
print(mlp)

# initialize optimizer and loss functions
opt = SGD(mlp.parameters(), lr=LR) # learning rate
lossFunc = nn.CrossEntropyLoss()

for epoch in range(0, EPOCHS):
    print("[INFO] epoch: {}".format(epoch))
    trainLoss = 0
    trainAcc = 0
    samples = 0
    mlp.train() # can update parameters of network, perform forward/bacward propagation
    # need to put model in training mode to update the parameters of model

    # loop over all the batches
    for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):
        '''
        batchX.shape = [BATCH_SIZE, num_features]
        batchX is an array with BATCH_SIZE's samples. And each sample with have 'num_features' size
        '''
        (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE)) # move the batches to the devices that will process them
        predictions = mlp(batchX) # make the prediction
        loss = lossFunc(predictions, batchY.long()) # compare our predictions with ground truth, its loss

        opt.zero_grad() # zero all the attributes '.grad' to zero
        # backward() updates '.grad' of mlp that shows the gradients of the model
        loss.backward() # doing backprop, it will accumulate gradients at each step. If you do not...
        # zero the gradients done previously, these will start to accumulate
        opt.step() # update model parameters. Take step towards the optimal face

        # update training loss, accuracy, and the number of samples visited
        trainLoss += loss.item() * batchY.size(0) # multiply by size because loss.item() returns the average loss...
        # across all the samples. and you want the total accumulative loss
        trainAcc += (predictions.max(1)[1] == batchY).sum().item() # sum tells num of True values, item converts to integer
        '''
        ** Assume that we had a batch size of 3, where for sample0 you have value 0.8 (probability)...
        for the class (of the three options you have) with index 1. Sample2 give that class with max...
        probability is the one with index0 with prob 0.6, etc..
        values, indices = predictions.max(1)
            # values = [0.8, 0.6, 0.5]  # The maximum probability values
            # indices = [1, 0, 2]       # Which class had the max (the predictions!)
        predictions.max(1)[1] are the max probabilities gotten
        '''
        samples += batchY.size(0) # number of samples

    # display model progress
    trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
    print(trainTemplate.format(epoch, (trainLoss / samples), (trainAcc / samples)))

    ## Here we examine accuracy and loss of model
    testLoss = 0
    testAcc = 0
    samples = 0
    mlp.eval() # no gradient computation being done. not updating model parameters
    # need to put on evaluation mode if want to make predictions

    # initialize no gradient context, no doing gradient. because we are going to evaluate
    with torch.no_grad():
        for (batchX, batchY) in next_batch(testX, testY, BATCH_SIZE):
            (batchX, batchY) = (batchX.to(DEVICE),
                                batchY.to(DEVICE))  # move the batches to the devices that will process them

            predictions = mlp(batchX)  # make the prediction
            loss = lossFunc(predictions, batchY.long())  # get the loss
            testLoss += loss.item() * batchY.size(0)
            testAcc += (predictions.max(1)[1] == batchY).sum().item()
            samples += batchY.size(0)  # number of samples

    # display model progress
    testTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
    print(testTemplate.format(epoch, (testLoss / samples), (testAcc / samples)))

