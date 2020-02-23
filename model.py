import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size):

        super(MultiLayerPerceptron, self).__init__()

        ######

        # 4.3 YOUR CODE HERE
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.under_fitting = nn.Linear(input_size, 1)
        self.over_fitting = nn.Linear(64, 1)
        ######

    def forward(self, features):

        ######

        # 4.3 YOUR CODE HERE
        # x = self.under_fitting(features)

        x = self.fc1(features)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.over_fitting(x)
        x = torch.sigmoid(x).reshape(-1)

        return x
        ######
