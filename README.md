# DigitRecognition

# Update
**User interface still in development!!!**

# Description

A fully connected neural network recognizing hand-written digits with **NumPy**.

Use **MNIST** `.csv` dataset to train. (Ignored in the repository)

Use **Pygame** to visualize the process and implement the drawing pad interface.

# Installation
* NumPy: `pip install numpy`
* Pygame: `pip install pygame`

* Update the `MNIST_path` in `NeuralNetwork.py` to the corresponding training dataset `.csv` file.

    `MNIST_path = './MNIST/mnist_test.csv'`

# Usage
Run `NeuralNetwork.py`

# Code
* `NeuralNetwork.py`

    Main code file.

    **Pygame** running code.
* `Layers.py`

    Class `Hidden_Layer`, `Output_Layer`

    Include `.forward()`, `.backward()`, `.learn()` method for forward propagation, backward propagation, adjusting weight & bias.

* `ActivationFunction.py`

    **Activation functions**: *ReLU*, *Sigmoid*, *tanh*, *Softmax*.

    **Loss function**: *Cross Entropy*

    and their derivatives.
