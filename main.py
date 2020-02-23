import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *

from scipy.signal import savgol_filter


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0

# =================================== LOAD DATASET =========================================== #

######

# 3.1 YOUR CODE HERE

data = pd.read_csv("data/adult.csv")



######

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

######

# 3.2 YOUR CODE HERE

data_shape = data.shape
data_columns = data.columns
data_head = data.head()
value_counts = data['income'].value_counts()


print("data shape: ")
print(data_shape)
print("\n")
print("data columns: ")
print(data_columns)
print("\n")
print("data_head: ")
verbose_print(data_head)
print('\n')
print("value_counts: ")
print(value_counts)
print("\n")
######


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]

for feature in col_names:
    ###
    print(feature)
    print(data[feature].isin(["?"]).sum())
    print("\n")

    # 3.3 YOUR CODE HERE

    ######

# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

    ######

    # 3.3 YOUR CODE HERE

    try:
        data = data[data[feature] != "?"]
    except ValueError:
        continue

print("Number of rows after clean up: %s" % data.shape[0])
    ######

# =================================== BALANCE DATASET =========================================== #

    ######

    # 3.4 YOUR CODE HERE
n = min(value_counts[0], value_counts[1])
low_income_data = data[data['income'] != ">50K"]
high_income_data = data[data['income'] != "<=50K"]
low_income_data = low_income_data.sample(n=n, random_state=1)
high_income_data = high_income_data.sample(n=n, replace=True, random_state=1)
data = pd.concat([low_income_data, high_income_data], ignore_index=True, sort=False)
data = data.sample(frac=1, random_state=400).reset_index(drop=True)


    ######

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######

# 3.5 YOUR CODE HERE

data_desc = data.describe()
verbose_print(data_desc)

cate_dict = {}

######

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    ######

    # 3.5 YOUR CODE HERE
    col = data[feature]
    distinct_val = col.unique()
    cate_dict[feature] = distinct_val

for feature, val in cate_dict.items():
    print("Categorical Feature: %s" % feature)
    for y in val:
        num = data[feature].isin([y]).sum()
        print("%s: %s" % (y, num))
    print("\n")




    ######

# visualize the first 3 features using pie and bar graphs

######

# 3.5 YOUR CODE HERE
    # pie_chart(data, feature)
    # binary_bar_chart(data, feature)

######

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# NORMALIZE CONTINUOUS FEATURES
######

# 3.6 YOUR CODE HERE
continuous_feats = set(data.columns.tolist()).difference(categorical_feats)
continuous_data = data[continuous_feats]

for feature in continuous_feats:
    mean = data[feature].mean()
    std = data[feature].std()
    continuous_data[feature] = (continuous_data[feature] - mean)/std

######

# ENCODE CATEGORICAL FEATURES
label_encoder = LabelEncoder()

######

# 3.6 YOUR CODE HERE
categorical_data = data[categorical_feats]
for feature in categorical_feats:
    categorical_data[feature] = label_encoder.fit_transform(categorical_data[feature])

######

oneh_encoder = OneHotEncoder()
######

# 3.6 YOUR CODE HERE

label = categorical_data['income'].values

categorical_data = categorical_data.drop('income', 1)
categorical_data = pd.DataFrame(oneh_encoder.fit_transform(categorical_data))

entire_data = pd.concat([categorical_data, continuous_data], axis=1)
entire_data = entire_data.values

final_data = np.zeros((entire_data.shape[0], 103))
i = 0
for x in entire_data:
    final_data[i, :] = np.concatenate((x[0].toarray().reshape(97, ), x[1: 7]), axis=0).astype(np.float64)
    i += 1

######
# Hint: .toarray() converts the DataFrame to a numpy array



# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######

# 3.7 YOUR CODE HERE

trainData, valData, trainLabel, valLabel = train_test_split(final_data, label, test_size=0.2, random_state=1, shuffle=True)

######

# =================================== LOAD DATA AND MODEL =========================================== #


def load_data(batch_size):
    ######

    # 4.1 YOUR CODE HERE

    trainDataset = AdultDataset(trainData, trainLabel)
    valDataset = AdultDataset(valData, valLabel)

    train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=valDataset, batch_size=batch_size, shuffle=True)

    ######

    return train_loader, val_loader


def load_model(lr):

    ######

    # 4.4 YOUR CODE HERE

    model = MultiLayerPerceptron(input_size=103)

    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)


    ######

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    ######

    # 4.6 YOUR CODE HERE

    for j, data in enumerate(val_loader, 0):
        inputs, val_labels = data
        predicts = model(inputs.float())

        for k in range(len(val_labels)):
            if predicts[k] > 0.5:
                temp = 1.0
            else:
                temp = 0.0
            if temp == val_labels[k]:
                total_corr += 1.0
    ######

    return float(total_corr)/len(val_loader.dataset)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)

    args = parser.parse_args()

    ######

    # 4.5 YOUR CODE HERE

    trainLoss = []
    trainAcc = []
    valAcc = []
    trainTime = []

    train_loader, val_loader = load_data(args.batch_size)
    MLP, loss_func, optimizer = load_model(args.lr)

    a = time()
    for i in range(args.epochs):
        running_loss = 0.0
        total_corr = 0.0
        total_data = 0.0
        step_corr = 0.0
        for j, data in enumerate(train_loader, 0):
            inputs, train_labels = data

            optimizer.zero_grad()

            predicts = MLP(inputs.float())
            train_loss = loss_func(input=predicts.squeeze(), target=train_labels.float())
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()
            for k in range(len(train_labels)):
                if predicts[k] > 0.5:
                    temp = 1.0
                else:
                    temp = 0.0
                if temp == train_labels[k]:
                    total_corr += 1.0
                    step_corr += 1.0

            total_data += len(train_labels)
            step_corr = 0.0

            if (j % args.eval_every) == 0:
                running_loss /= args.eval_every
                total_corr /= total_data
                val_acc = evaluate(MLP, val_loader)
                train_time = time() - a

                print("iter: " + str(i) + " cost: " + str(running_loss) + " training accuracy: " + str(total_corr) + " validation acc: " + str(val_acc))

                trainLoss.append(running_loss)
                trainAcc.append(total_corr)
                valAcc.append(val_acc)
                trainTime.append(train_time)

                running_loss = 0.0
                total_corr = 0.0
                total_data = 0.0

    b = time()

    diff = b - a

    print("time training loop take is : %s" % diff)

    smooth_train_acc = savgol_filter(trainAcc, 55, 2)
    smooth_val_acc = savgol_filter(valAcc, 55, 2)
    # smooth_train_acc = trainAcc
    # smooth_val_acc = valAcc

    '''
    tmp_train = np.array(trainAcc)
    tmp_val = np.array(valAcc)
    np.save("train_acc_tanh.npy", tmp_train)
    np.save("val_acc_tanh.npy", tmp_val)
    '''

    plt.figure()
    plt.title("Training and Validation Accuracies vs. Number of Gradient Steps")
    plt.plot(np.array(np.arange(len(smooth_train_acc))), smooth_train_acc, color='orange', label='training')
    plt.plot(np.array(np.arange(len(smooth_val_acc))), smooth_val_acc, color='blue', label='validation')
    plt.xlabel("Number of Steps")
    plt.ylabel("Accuracies")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Training and Validation Accuracies vs. Time")
    plt.plot(trainTime, smooth_train_acc, color='orange', label='training')
    plt.plot(trainTime, smooth_val_acc, color='blue', label='validation')
    plt.xlabel("Train Time")
    plt.ylabel("Accuracies")
    plt.legend()
    plt.show()

        ######


if __name__ == "__main__":
    main()
