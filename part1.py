import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


def load_data(type, delimiter=","):
    data_name = '{}data.csv'.format(type)
    label_name = '{}label.csv'.format(type)
    data = np.loadtxt(data_name, dtype=np.single, delimiter=delimiter)
    label = np.loadtxt(label_name, dtype=np.single, delimiter=delimiter)

    return data, label


class SNC(nn.Module):
    def __init__(self, af):
        super(SNC, self).__init__()
        self.fc1 = nn.Linear(9, 1)
        self.af = af

    def forward(self, I):
        x = self.fc1(I)
        if self.af == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.af == 'relu':
            x = F.relu(x)
        return x


def accuracy(predictions, label):
    num_val = 0
    for k in range(len(label)):
        if predictions[k] > 0.5:
            predictions[k] = 1.0
        else:
            predictions[k] = 0.0
        if predictions[k] == label[k]:
            num_val += 1.0
    accuracy = num_val / len(label)

    return accuracy


def plot_graph(title_name, x_label_name, y_label_name, trainData, validData):
    plt.figure()
    plt.title(title_name)
    plt.plot(np.array(np.arange(len(trainData))), trainData, color='orange', label='training')
    plt.plot(np.array(np.arange(len(validData))), validData, color='blue', label='validation')
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.legend()
    plt.show()


def grad_descent(trainData, trainLabel, valData, valLabel, lr, numEpoch, acttype):
    trainLoss = []
    valLoss = []
    trainAcc = []
    valAcc = []

    smallNN = SNC(acttype)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(smallNN.parameters(), lr=lr)

    for i in range(numEpoch):
        optimizer.zero_grad()
        predict = smallNN(trainData)
        train_loss = loss_function(input=predict.squeeze(), target=trainLabel.float())
        train_loss.backward()
        optimizer.step()
        train_acc = accuracy(predict, trainLabel)

        predict = smallNN(valData)
        val_loss = loss_function(input=predict.squeeze(), target=valLabel.float())
        val_acc = accuracy(predict, valLabel)

        trainLoss.append(train_loss)
        valLoss.append(val_loss)
        trainAcc.append(train_acc)
        valAcc.append(val_acc)

        print("iter: " + str(i) + " cost: " + str(train_loss.data) + " training accuracy: " + str(train_acc) + " validation accuracy: " + str(val_acc))

    return trainLoss, valLoss, trainAcc, valAcc


def main(args):
    torch.manual_seed(args.seed)
    train_data, train_label = load_data(args.trainingfile)
    val_data, val_label = load_data(args.validationfile)

    trainData = torch.from_numpy(train_data)
    trainLabel = torch.from_numpy(train_label)
    valData = torch.from_numpy(val_data)
    valLabel = torch.from_numpy(val_label)

    trainLoss, valLoss, trainAcc, valAcc = grad_descent(trainData, trainLabel, valData, valLabel, args.learningrate, args.numepoch, args.actfunction)

    plot_graph("Training and Validation Loss Curve", "Number of Epochs", "Training and Validation Loss", trainLoss,
               valLoss)
    plot_graph("Training and Validation Accuracy Curve", "Number of Epochs", "Training and Validation Accuracy",
               trainAcc, valAcc)



if __name__ == '__main__':
    # Command Line Arguments

    parser = argparse.ArgumentParser(description='generate training and validation data for assignment 2')
    parser.add_argument('--trainingfile', help='name stub for training data and label output in csv format',
                        default="train")
    parser.add_argument('--validationfile', help='name stub for validation data and label output in csv format',
                        default="valid")
    parser.add_argument('--numtrain', help='number of training samples', type=int, default=200)
    parser.add_argument('--numvalid', help='number of validation samples', type=int, default=20)
    parser.add_argument('--seed', help='random seed', type=int, default=396)
    parser.add_argument('--learningrate', help='learning rate', type=float, default=0.06)
    parser.add_argument('--actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'],
                        default='linear')
    parser.add_argument('--numepoch', help='number of epochs', type=int, default=100)

    args = parser.parse_args()

    traindataname = args.trainingfile + "data.csv"
    trainlabelname = args.trainingfile + "label.csv"

    print("training data file name: ", traindataname)
    print("training label file name: ", trainlabelname)

    validdataname = args.validationfile + "data.csv"
    validlabelname = args.validationfile + "label.csv"

    print("validation data file name: ", validdataname)
    print("validation label file name: ", validlabelname)

    print("number of training samples = ", args.numtrain)
    print("number of validation samples = ", args.numvalid)

    print("learning rate = ", args.learningrate)
    print("number of epoch = ", args.numepoch)

    print("activation function is ", args.actfunction)

    main(args)