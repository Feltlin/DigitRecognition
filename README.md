# DigitRecognition

# Update
**User interface still in development!!!**

# Description

A fully connected neural network recognizing hand-written digits with **NumPy**.

Use **MNIST** `.csv` dataset to train. (Ignored in the repository)

Use **Pygame** to visualize the process and implement the drawing pad interface.

Use **Threading** to separate model calculation and screen update.

Use **Tkinter** to load and save the trained model.

Use **Pillow** to process image.

# Installation
* NumPy: `pip install numpy`
* Pygame: `pip install pygame`
* Tkinter: `pip install tk`
* Pillow: `pip install pillow`

* Update the `MNIST_path` in `NeuralNetwork.py` to the corresponding training dataset `.csv` file.

    `MNIST_path = './MNIST/mnist_test.csv'`

# Usage
Run `NeuralNetwork.py` in the terminal.

`python .\NeuralNetwork.py`

# Code
* `NeuralNetwork.py`

    Main code file.

    **Pygame** running code.
* `Layer.py`

    Class `Hidden_Layer`, `Output_Layer`

    Include `.forward()`, `.backward()`, `.learn()` method for forward propagation, backward propagation, adjusting weight & bias.

* `ActivationFunction.py`

    **Activation functions**: $ReLU$, $Sigmoid$, $\tanh$, $Softmax$.

    **Loss function**: $Cross-entropy$

    and their derivatives.

* `PygameClass.py`

    Class `PAINT`: Drawing canvas
    Class `TEXT`: Text box
    Class `BUTTON`: Clickable button


# Math
Math equations such as matrix cannot properly display on **GitHub**.

This README file is written in **VS Code**.

Please use **VS Code** or other Markdown reader to view.

* ## Prior Knowledge

    Vector & Matrix:
    * Matrix Multiplication
    * Transpose

    Multivariable Calculus:
    * Partial Derivative
    * Gradient

    One-Hot Encoding

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

    $a^T$: transpose

* ## Neural Network
    ### Forward Propagation
    #### Layer Input

    $a^I_{1 \times I} = \begin{bmatrix}
                        a^I_1 & a^I_2 & \cdots & a^I_I
                        \end{bmatrix}$

    #### Layer 1

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

    #### Layer 2

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

    #### Layer Output

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
        \end{bmatrix}$ (One-Hot Encoding)
    
    $l = -\sum^O_{j=1} y_j\ln a^O_j = -y_1ln a^O_1 - y_2ln a^O_2 - \cdots - y_Oln a^O_O$ (Cross-entropy)

    ### Backward Propagation
    #### Layer Output

    $\begin{align}
    \notag {\frac{\partial l} {\partial a^O}}_{1 \times O}
    & = & \begin{bmatrix} \frac{\partial l} {\partial a^O_1} & \frac{\partial l} {\partial a^O_2} & \cdots & \frac{\partial l} {\partial a^O_O} \end{bmatrix} \\
    \notag & = & \begin{bmatrix} -\frac{y_1} {a^O_1} & -\frac{y_2} {a^O_2} & \cdots & -\frac{y_O} {a^O_O} \end{bmatrix}
    \end{align}$

    This is a *Jacobian* matrix:

    $\begin{align} \notag
    {\frac{\partial a^O} {\partial z^O}}_{O \times O}
    & = & \begin{bmatrix}
    \frac{\partial a^O_1} {\partial z^O_1} & \frac{\partial a^O_1} {\partial z^O_2} & \cdots & \frac{\partial a^O_1} {\partial z^O_O} \\ 
    \frac{\partial a^O_2} {\partial z^O_1} & \frac{\partial a^O_2} {\partial z^O_2} & \cdots & \frac{\partial a^O_2} {\partial z^O_O} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial a^O_O} {\partial z^O_1} & \frac{\partial a^O_O} {\partial z^O_2} & \cdots & \frac{\partial a^O_O} {\partial z^O_O}
    \end{bmatrix} \\
    \notag & = & \begin{bmatrix}
    a^O_1(1-a^O_1) & -a^O_1a^O_2 & \cdots & -a^O_1a^O_O \\
    -a^O_1a^O_2 & a^O_2(1-a^O_2) & \cdots & -a^O_2a^O_O \\
    \vdots & \vdots & \ddots & \vdots \\
    -a^O_1a^O_O & -a^O_2a^O_O & \cdots & a^O_O(1-a^O_O)
    \end{bmatrix}
    \end{align}$

    $\because y$ is a *One-Hot*, $\sum^O_{j=1}y_j=1$

    $\therefore$

    $\begin{align}
    \notag {\frac{\partial l} {\partial z^O}}_{1 \times O}
    & = & {\frac{\partial l} {\partial a^O}}_{1 \times O} \cdot {\frac{\partial a^O} {\partial z^O}}_{O \times O} \\
    \notag & = & \begin{bmatrix} -y_1+a^O_1\sum^O_{j=1}y_j & -y_2+a^O_2\sum^O_{j=1}y_j & \cdots -y_O+a^O_1\sum^O_{j=1}y_j \end{bmatrix} \\
    \notag & = & \begin{bmatrix} a^O_1-y_1 & a^O_2-y_2 & \cdots a^O_O-y_O\end{bmatrix} \\
    \notag & = & a^O-y
    \end{align}$

    $\begin{align}
    \notag {\frac{\partial l} {\partial w^O}}_{k \times O}
    & = & {\frac{\partial z^O} {\partial w^O}}_{k \times 1} \cdot {\frac{\partial l^O} {\partial z^O}}_{O \times O} \\
    \notag & = & a^{2T} \cdot \frac{\partial l} {\partial z^O}
    \end{align}$

    $\because \frac{\partial z^O} {\partial b^O}$ is a *Jacobian* matrix and is a *identity* matrix

    $\therefore$

    $\begin{align}
    \notag {\frac{\partial l} {\partial b^O}}_{1 \times O}
    & = & {\frac{\partial l} {\partial z^O}}_{1 \times O} \cdot {\frac{\partial z^O} {\partial b^O}}_{O \times O} \\
    \notag & = & \frac{\partial l} {\partial z^O} \cdot 1
    \end{align}$

    #### Layer 2

    $\begin{align}
    \notag {\frac{\partial l} {\partial a^2}}_{1 \times k}
    & = & {\frac{\partial l} {\partial z^O}}_{1 \times O} \cdot {\frac{\partial z^O} {\partial a^2}}_{O \times k} \\
    \notag & = & \frac{\partial l} {\partial z^O} \cdot w^{OT}
    \end{align}$

    $\begin{align}
    \notag {\frac{\partial l} {\partial z^2}}_{1 \times k}
    & = & {\frac{\partial l} {\partial a^2}}_{1 \times k} \cdot {\frac{\partial a^2} {\partial z^2}}_{k \times k} \\
    \notag & = & \frac{\partial l} {\partial a^2} \circ ReLU'(z^2)
    \end{align}$

    $\begin{align}
    \notag {\frac{\partial l} {\partial w^2}}_{m \times k}
    & = & {\frac{\partial z^2} {\partial w^2}}_{m \times 1} \cdot {\frac{\partial l} {\partial z^2}}_{1 \times k} \\
    \notag & = & a^{1T} \cdot {\frac{\partial l} {\partial z^2}}
    \end{align}$

    $\because \frac{\partial z^2} {\partial b^2}$ is a *Jacobian* matrix and is a *identity* matrix

    $\therefore$

    $\begin{align}
    \notag {\frac{\partial l} {\partial b^2}}_{1 \times k}
    & = & {\frac{\partial l} {\partial z^2}}_{1 \times k} \cdot {\frac{\partial z^2} {\partial b^2}}_{k \times k} \\
    \notag & = & \frac{\partial l} {\partial z^2} \cdot 1
    \end{align}$

    #### Layer 1

    $\begin{align}
    \notag {\frac{\partial l} {\partial a^1}}_{1 \times m}
    & = & {\frac{\partial l} {\partial z^2}}_{1 \times k} \cdot {\frac{\partial z^2} {\partial a^1}}_{k \times m} \\
    \notag & = & \frac{\partial l} {\partial z^2} \cdot w^{2T}
    \end{align}$

    $\begin{align}
    \notag {\frac{\partial l} {\partial z^1}}_{1 \times m}
    & = & {\frac{\partial l} {\partial a^1}}_{1 \times m} \cdot {\frac{\partial a^1} {\partial z^1}}_{m \times m} \\
    \notag & = & \frac{\partial l} {\partial a^1} \circ ReLU'(z^1)
    \end{align}$

    $\begin{align}
    \notag {\frac{\partial l} {\partial w^1}}_{I \times m}
    & = & {\frac{\partial z^1} {\partial w^1}}_{I \times 1} \cdot {\frac{\partial l} {\partial z^1}}_{1 \times m} \\
    \notag & = & a^{IT} \cdot {\frac{\partial l} {\partial z^1}}
    \end{align}$

    $\because \frac{\partial z^1} {\partial b^1}$ is a *Jacobian* matrix and is a *identity* matrix

    $\therefore$

    $\begin{align}
    \notag {\frac{\partial l} {\partial b^1}}_{1 \times m}
    & = & {\frac{\partial l} {\partial z^1}}_{1 \times m} \cdot {\frac{\partial z^1} {\partial b^1}}_{m \times m} \\
    \notag & = & \frac{\partial l} {\partial z^1} \cdot 1
    \end{align}$