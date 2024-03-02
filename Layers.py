import numpy as np
from ActivationFunction import *
#Input Layer
class Hidden_Layer:

    #Initialize function
    def __init__(self,n_input:int,n_neuron:int):
        self.w = np.random.randn(n_input,n_neuron) * 1e-2
        self.dw_temp = np.zeros((n_input,n_neuron))

        self.b = np.zeros((1,n_neuron))
        self.db_temp = np.zeros((1,n_neuron))

    #Output for the next layer's input.
    def forward(self,input):
        self.input = input
        self.z = (input @ self.w) + self.b
        self.output = relu(self.z)
    
    #Back propagation of the hidden layer.
    def backward(self,doutput):
        self.dz = doutput * d_relu(self.z)
        self.dw = (self.input.T @ self.dz) / 784
        self.db = self.dz / 784
        self.dinput = self.dz @ self.w.T
        self.dw_temp += self.dw
        self.db_temp += self.db

    #Adjust the weight and bias to learn.
    def learn(self,rate,batch_size):
        self.w -= rate * self.dw_temp / (np.max(np.abs(self.dw_temp)) * batch_size)
        self.b -= rate * self.db_temp / (np.max(np.abs(self.db_temp)) * batch_size)
        self.dw_temp.fill(0)
        self.db_temp.fill(0)

#Output Layer
class Output_Layer:

    #Initialize function
    def __init__(self,n_input:int,n_output:int):
        self.w = np.random.randn(n_input,n_output) * 1e-2
        self.dw_temp = np.zeros((n_input,n_output))

        self.b = np.zeros((1, n_output))
        self.db_temp = np.zeros((1, n_output))
    
    #Output for the final prediction.
    def forward(self,input):
        self.input = input
        self.z = (input @ self.w) + self.b
        self.output = softmax(self.z)

    #Back propagation of the output layer.
    def backward(self,desire_output):
        self.dz = self.output - desire_output
        self.dw = (self.input.T @ self.dz) / 784
        self.db = self.dz / 784
        self.dinput = self.dz @ self.w.T
        self.dw_temp += self.dw
        self.db_temp += self.db
    
    #Adjust the weight and bias to learn.
    def learn(self,rate:float,batch_size:int):
        self.w -= rate * self.dw_temp / (np.max(np.abs(self.dw_temp)) * batch_size)
        self.b -= rate * self.db_temp / (np.max(np.abs(self.db_temp)) * batch_size)
        self.dw_temp.fill(0)
        self.db_temp.fill(0)