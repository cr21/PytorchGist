# usage python train.py --model output/model.pth --plot output/plot.png
import matplotlib

# set backend option so figure can be saved in background
matplotlib.use("agg")

from model.lenet import Lenet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

# configure argument

arparser = argparse.ArgumentParser()
arparser.add_argument("-m", "--model", type=str, required=True, help="Path to Output trained model")
arparser.add_argument("-p", "--plot", type=str, required=True, help="Path to save train/test  loss and accuracy plot")
args = vars(arparser.parse_args())

LR_INIT = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATA SET
print("[INFO] Loading dataset")
# use cached data from data folder if data folder is empty download and put it inside data folder
trainData = KMNIST(root="data", train=True, download=True, transform=ToTensor())
testData = KMNIST(root="data", train=False, download=True, transform=ToTensor())

# create Validation data from train data
print("[INFO] create validation data from train data")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, validationData) = random_split(trainData, [numTrainSamples, numValSamples],
                                           generator=torch.Generator().manual_seed(42))

# initialize Train, Test and Validation DataLoader
trainDataLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
valDataLoader = DataLoader(dataset=validationData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE)

trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

# Initialize LENET model
print("[INFO] initialize LENET MODEL")
lenetModel = Lenet(numChannels=1, outclasses=len(trainData.dataset.classes)).to(device)

# optimizer and LOSS

opt = Adam(lenetModel.parameters(), lr=LR_INIT)
lossFunc = nn.NLLLoss()

# initialize the dictonary to store training Histort

H = {

    "trainLoss": [],
    "trainAcc": [],
    "valLoss": [],
    "valAcc": []
}

print("[INFO] start training network")
startTime = time.time()

for epoch in range(0, EPOCHS):

    # train loop
    lenetModel.train()

    totalTrainLoss = 0
    totalValLoss = 0

    trainCorrect = 0
    valCorrect = 0

    # training loop
    for (xTrain, yTrain) in trainDataLoader:
        (xTrain, yTrain) = (xTrain.to(device), yTrain.to(device))

        # prediction
        predictions = lenetModel(xTrain)

        # print(predictions)
        loss = lossFunc(predictions, yTrain)

        # zero out the previously accumulated gradient
        opt.zero_grad()
        # back propagate
        loss.backward()
        # update the gradient
        opt.step()

        # update the total trainLoss and total Train and validation accuracy

        totalTrainLoss += loss
        trainCorrect += (predictions.argmax(1) == yTrain).type(torch.float).sum().item()


    # validation loop
    with torch.no_grad():
        lenetModel.eval()
        for (xVal, yVal) in valDataLoader:
            (xVal, yVal) = (xVal.to(device), yVal.to(device))

            predictions = lenetModel(xVal)
            totalValLoss += lossFunc(predictions, yVal)

            valCorrect += (predictions.argmax(1) == yVal).type(torch.float).sum().item()

    avgTrainingLoss = totalTrainLoss / trainSteps
    avgValidationLoss = totalValLoss / valSteps

    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)

    H["trainLoss"].append(avgTrainingLoss.cpu().detach().numpy())
    H["valLoss"].append(avgValidationLoss.cpu().detach().numpy())
    H["trainAcc"].append(trainCorrect)
    H["valAcc"].append(valCorrect)

    # print model training and validation information

    print("[INFO] EPOCH : {}/{} ".format(epoch, EPOCHS))
    print("[INFO] Training Loss : {:.6f}, Train Accuracy : {:.4f}".format(avgTrainingLoss, trainCorrect))
    print("[INFO] Validation Loss : {:.6f}, Validation Accuracy : {:.4f}".format(avgValidationLoss, valCorrect))

endTime = time.time()
print("[INFO] Time Taken to train the model : {:.2f} ".format(endTime - startTime))

print("[INFO] Evaluating model using testData")


with torch.no_grad():
    lenetModel.eval()
    testPredictions = []
    for (xTest, yTest) in testDataLoader:
        xTest = xTest.to(device)
        prediction = lenetModel(xTest)
        testPredictions.extend(prediction.argmax(axis=1).cpu().numpy())

# generate Classification report

print(classification_report(testData.targets.cpu().numpy(), np.array(testPredictions), target_names=testData.classes))

# we can now evaluate the network on the test set

# plotting model

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["trainLoss"], label="train_loss")
plt.plot(H["valLoss"], label="val_loss")
plt.plot(H["trainAcc"], label="train_acc")
plt.plot(H["valAcc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
torch.save(lenetModel, args["model"])
