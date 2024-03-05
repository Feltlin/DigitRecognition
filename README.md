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

    **Activation functions**: $ReLU$, $Sigmoid$, $\tanh$, $Softmax$.

    **Loss function**: $Cross-entropy$

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
                \begin{matrix}
                1 & x>0 \\
                0 & x<0
                \end{matrix}
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

    $softmax(x)_i = \frac{e^{x_i}} {\sum^K_{j=1} e^{x_j}}$

    Derivative of softmax:

    Jacobian matrix(To be updated...)

* ## Loss Function

    Cross-entropy:

    $H(p,q)=-\sum p(x)\log q(x)$
    <br><br>

    Mean squared error:

    $MSE=\frac{1} {n} \sum^n_{i=1} (y_i-\hat{y}_i)^2$

    $y$: desired value

    $\hat{y}$: predicted value

* ## Symbol Notation

    $w\cdot b$: dot product / matrix multiplication

    $w \circ b$: Hadamard product / element-wise product

    $k \times j$: matrix / vector dimension, $k$ rows, $j$ columns

* ## Neural Network

    ### Forward Propagation

    #### Layer Input:

    $a^I_{1 \times I} = \begin{bmatrix}
                        a^I_1 & a^I_2 & \cdots & a^I_I
                        \end{bmatrix}$

    #### Layer 1:

    $w^1_{I \times m} = \begin{bmatrix}
                        w^1_{1,1} & w^1_{1,2} & \cdots & w^1_{1,m} \\
                        w^1_{2,1} & w^1_{2,2} & \cdots & w^1_{2,m} \\
                        \vdots & \vdots & \ddots & \vdots \\
                        w^1_{I,1} & w^1_{I,2} & \cdots & w^1_{I,m}
                        \end{bmatrix}$

    $b^1_{1 \times m} = \begin{bmatrix}
                        b^1_1 & b^1_2 & \cdots & b^1_m
                        \end{bmatrix}$

    $z^1_{1 \times m} = a^I_{1 \times I} \cdot w^1_{I \times m} + b^1_{1 \times m}$

    $a^1_{1 \times m} = ReLU(z^1_{1 \times m})$

    #### Layer 2:

    $w^2_{m \times k} = \begin{bmatrix}
                        w^2_{1,1} & w^2_{1,2} & \cdots & w^2_{1,k} \\
                        w^2_{2,1} & w^2_{2,2} & \cdots & w^2_{2,k} \\
                        \vdots & \vdots & \ddots & \vdots \\
                        w^2_{m,1} & w^2_{m,2} & \cdots & w^2_{m,k}
                        \end{bmatrix}$

    $b^2_{1 \times k} = \begin{bmatrix}
                        b^2_1 & b^2_2 & \cdots & b^2_k
                        \end{bmatrix}$

    $z^2_{1 \times k} = a^1_{1 \times m} \cdot w^2_{m \times k} + b^2_{1 \times k}$

    $a^2_{1 \times k} = ReLU(z^2_{1 \times k})$

    #### Layer Output:

    $w^O_{k \times O} = \begin{bmatrix}
                        w^O_{1,1} & w^O_{1,2} & \cdots & w^O_{1,O} \\
                        w^O_{2,1} & w^O_{2,2} & \cdots & w^O_{2,O} \\
                        \vdots & \vdots & \ddots & \vdots \\
                        w^O_{k,1} & w^O_{k,2} & \cdots & w^O_{k,O}
                        \end{bmatrix}$

    $b^O_{1 \times O} = \begin{bmatrix}
                        b^O_1 & b^O_2 & \cdots & b^O_O
                        \end{bmatrix}$

    $z^O_{1 \times O} = a^2_{1 \times k} \cdot w^O_{k \times O} + b^O_{1 \times O}$

    $a^O_{1 \times O} = softmax(z^O_{1 \times O})$

    #### Loss

    $y = \begin{bmatrix}
        y_1 & y_2 & \cdots & y_O
        \end{bmatrix}$
    
    $l = -\sum^O_{j=1} y_j\ln a^O_j = -y_1ln a^O_1 - y_2ln a^O_2 - \cdots - y_Oln a^O_O$