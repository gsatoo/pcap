# cedro blog
# PyTorch まずMLPを使ってみる
# http://cedro3.com/ai/pytorch-mlp/
#

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms


# Multi Layer Perceptron Network
class MLPNet (nn.Module):
    def __init__(self,
                 numOfFeatures,
                 numOfLabels,
                 numOfNeurons=32):
        super(MLPNet, self).__init__()


        #
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(numOfFeatures, numOfNeurons),
            nn.BatchNorm1d(numOfNeurons), ######
            nn.ReLU(),
            # nn.Dropout2d(0.2), ###
            nn.Linear(numOfNeurons, numOfNeurons),
            nn.BatchNorm1d(numOfNeurons), ######
            nn.ReLU(),
            # nn.Dropout2d(0.2), ###
            nn.Linear(numOfNeurons, numOfLabels)
            # nn.ReLU() ###
            )
        #

        # self.fc1 = nn.Linear(numOfFeatures, numOfNeurons)   
        # self.fc2 = nn.Linear(numOfNeurons, numOfNeurons)
        # self.fc3 = nn.Linear(numOfNeurons, numOfLabels)
        # self.dropout1 = nn.Dropout2d(0.2)
        # self.dropout2 = nn.Dropout2d(0.2)

        # #
        # print(f"# num Of Features is {numOfFeatures}.")
        # print(f"# num Of Labels is {numOfLabels}.")
        
    def forward(self, x):

        #
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        #

        # x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        # return F.relu(self.fc3(x))

