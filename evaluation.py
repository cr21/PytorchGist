# predict from loaded model

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import numpy as np
import argparse
import torch
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="Path to trained model")
args = vars(ap.parse_args())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] loading test dataset")
testData = KMNIST(root="data", train=False, download=True, transform=ToTensor())
idxs = np.random.choice(range(0, len(testData)), size=(10,))
testData = Subset(testData, idxs)
testDataLoader = DataLoader(testData, batch_size=1)

# load the model from file and set it to evaluation mode
lenetModel = torch.load(args["model"]).to(device)
lenetModel.eval()


#switch off the gradient
count = 1;
with torch.no_grad():
    for(image, label) in testDataLoader:
        print("SHAPE", image.numpy().shape)
        origImage = image.numpy().squeeze(axis=(0,1))
        origLabel = testData.dataset.classes[label.numpy()[0]]

        #send input to device and make prediction

        image = image.to(device)
        pred = lenetModel(image)

        # find the class label index with largest probability
        idx = pred.argmax(1).cpu().numpy()[0]
        predLabel = testData.dataset.classes[idx]


        # convert Image to gray scale to RGB scale so that we can write on image

        origImage = np.dstack([origImage]*3)
        origImage = imutils.resize(origImage, width=128)

        # draw predicted class label on top of image
        color = (0,255,0) if origLabel == predLabel else (0,0,255)
        cv2.putText(origImage,origLabel,(2,25),cv2.FONT_HERSHEY_SIMPLEX, 0.95,color,2)

        print("[INFO] True Label : {} , predicted Label : {} ".format(
            origLabel, predLabel))
        cv2.imshow("image" + str(count),origImage)
        cv2.waitKey(0)
        cv2.imwrite("output/image" + str(count) + ".jpeg", origImage)
        count += 1

