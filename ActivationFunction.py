import numpy as np

#ReLU (Rectified Linear Unit) function
def relu(input):
    return np.maximum(0.001,input)

#Derivative of ReLU function
def d_relu(input):
    input[input > 0] = 1
    input[input < 0] = 0
    return input

#Sigmoid function
def sigmoid(input):
    return 1 / (1 + np.exp(-input))

#Derivative of Sigmoid function
def d_sigmoid(input):
    return sigmoid(input) * (1 - sigmoid(input))

#Tanh function
def tanh(input):
    return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))

#Derivative of Tanh function
def d_tanh(input):
    return 1 - np.power(tanh(input),2)

#Softmax function
def softmax(input):
    exp = np.exp(input - np.max(input))
    return exp / np.sum(exp)

#Derivative of Softmax function
def d_softmax(input):
    input = input.reshape(-1)
    jacobian = np.zeros((len(input), len(input)))
    for i in range(len(input)):
        for j in range(len(input)):
            if i != j:
                jacobian[i, j] = -input[j] * input[i]
            else:
                jacobian[i, j] = input[j] * (1 - input[j])
    return jacobian

#Cross Entropy (Calculate the loss of this current neural network.)
def cross_entropy(layer_output,desire_output):
    return -np.sum(desire_output * np.log(layer_output + 1e-16))