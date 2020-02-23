import numpy as np
import matplotlib.pyplot as plt

trainAcc_relu = np.load("train_acc_relu.npy")
trainAcc_tanh = np.load("train_acc_tanh.npy")
trainAcc_sigmoid = np.load("train_acc_sigmoid.npy")
valAcc_relu = np.load("val_acc_relu.npy")
valAcc_tanh = np.load("val_acc_tanh.npy")
valAcc_sigmoid = np.load("val_acc_sigmoid.npy")

plt.figure()
plt.title("Training and Validation Accuracies for Relu, Sigmoid, Tanh Activation vs. Number of Gradient Steps")
plt.plot(np.array(np.arange(len(trainAcc_relu))), trainAcc_relu, label='Relu training')
plt.plot(np.array(np.arange(len(valAcc_relu))), valAcc_relu, label='Relu validation')
plt.plot(np.array(np.arange(len(trainAcc_relu))), trainAcc_tanh, label='Tanh training')
plt.plot(np.array(np.arange(len(valAcc_relu))), valAcc_tanh, label='Tanh validation')
plt.plot(np.array(np.arange(len(trainAcc_relu))), trainAcc_sigmoid, label='Sigmoid training')
plt.plot(np.array(np.arange(len(valAcc_relu))), valAcc_sigmoid, label='Sigmoid validation')
plt.xlabel("Number of Steps")
plt.ylabel("Accuracies")
plt.legend()
plt.show()