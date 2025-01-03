import numpy as np
from ActivationFunction import *
#Input Layer
class Hidden_Layer:

    #Initialize function
    def __init__(self, n_input:int, n_neuron:int):
        self.w = np.random.randn(n_input, n_neuron) * 1e-2
        self.dw_temp = np.zeros((n_input, n_neuron))

        self.b = np.zeros((1, n_neuron))
        self.db_temp = np.zeros((1, n_neuron))

    #Output for the next layer's input.
    def forward(self, input):
        self.input = input
        self.z = (input @ self.w) + self.b
        self.output = relu(self.z)
    
    #Back propagation of the hidden layer.
    def backward(self, doutput):
        self.dz = doutput * d_relu(self.z)
        self.dw = (self.input.T @ self.dz) / 784
        self.db = self.dz / 784
        self.dinput = self.dz @ self.w.T
        self.dw_temp += self.dw
        self.db_temp += self.db

    #Adjust the weight and bias to learn.
    def learn(self, rate:float, batch_size:int):
        self.w -= rate * self.dw_temp / (np.max(np.abs(self.dw_temp)) * batch_size)
        self.b -= rate * self.db_temp / (np.max(np.abs(self.db_temp)) * batch_size)
        self.dw_temp.fill(0)
        self.db_temp.fill(0)

#Output Layer
class Output_Layer(Hidden_Layer):

    def __init__(self, n_input:int, n_output:int):
        #Inherit from Hidden_Layer.
        super().__init__(n_input, n_output)
    
    def forward(self, input):
        self.input = input
        self.z = (input @ self.w) + self.b
        #Change activation function to softmax.
        self.output = softmax(self.z)

    def backward(self, desire_output):
        #Change ∂l/∂z.
        self.dz = self.output - desire_output
        self.dw = (self.input.T @ self.dz) / 784
        self.db = self.dz / 784
        self.dinput = self.dz @ self.w.T
        self.dw_temp += self.dw
        self.db_temp += self.db

#Convolutional Layer
class Convolutional_Layer:

    def __init__(self, kernel:np.ndarray):
        self.kernel = kernel
        self.flipped_kernel = np.flip(kernel)
        #Dimension of kernel.
        self.kernel_row, self.kernel_column = self.kernel.shape

    def forward(self, input:np.ndarray):
        self.input = input
        #Dimension of input.
        self.input_row, self.input_column = self.input.shape
        #Dimensions of output.
        self.output = np.zeros(self.input)
        pad_row = (self.kernel_row - 1) // 2
        pad_column = (self.kernel_column - 1) // 2
        padded_input = np.pad(self.input, pad_width = ((pad_row, pad_row), (pad_column, pad_column)), mode = "constant")
        #Convolution
        for i in range(self.input_row):
            for j in range(self.input_column):
                region = padded_input[i:i + self.kernel_row, j:j + self.kernel_column]
                self.output[i, j] = np.sum(region * self.flipped_kernel)
    
    def backward(self, doutput):
        ...
