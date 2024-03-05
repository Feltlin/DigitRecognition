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

# Math

* ## Variable

    $w$: weight

    $b$: bias

    $a$: activation

    $z$: unormalized activation (weighted sum)

    $L$: output layer

    $y$: desired output

    $l$: loss

* ## Activation Function

    Rectified linear unit:

    $ReLU(x) = \left\{
                \begin{matrix}
                x & x>0 \\
                0 & x\leqslant 0
                \end{matrix}
                \right.$

    Derivative of ReLU:

    $ReLU'(x) =\left\{
                \begin{matrix*}
                1 & x>0 \\
                0 & x<0
                \end{matrix*}
                \right.$
    <br><br>

    Sigmoid:

    $\sigma(x) = \frac{1}{1+e^{-x}}$

    Derivative of sigmoid:

    $\sigma'(x) = \sigma(x)(1-\sigma(x)) = \frac{1}{1+e^{-x}} (1-\frac{1}{1+e^{-x}})$
    <br><br>

    $\tanh$:

    $\tanh(x) = \frac{e^x-e^{-x}} {e^x+e^{-x}}$

    Derivative of $\tanh$

    $\tanh'(x) = 1-\tanh^2(x)$
    <br><br>

    Softmax:

    $softmax(x)_i = \frac{e^{x_i}} {\sum^K_{j=1}e^{x_j}}$

    Derivative of softmax:

    Jacobian matrix(To be updated...)

* ## Loss Function

    Cross-entropy:

    $H(p,q)=-\sum p(x)\log q(x)$
    <br><br>

    Mean squared error:

    $MSE=\frac{1} {n} \sum^n_{i=1}(y_i-\hat{y}_i)^2$

    $y$: desired value

    $\hat{y}$: predicted value

* ## Symbol Notation
    $w\cdot b$: dot product / matrix multiplication

    $w \circ b$: Hadamard product / element-wise product

