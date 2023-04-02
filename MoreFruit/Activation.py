import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    z = sigmoid(x)
    return np.multiply(z, (1 - z))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    z = tanh(x)
    return 1 - z ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1. * (x > 0)

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)

def softmax_derivative(x):
    z = softmax(x)
    return z * (1 - z)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def map_activaion(activation):
    if activation == "sigmoid":
        return sigmoid
    elif activation == "tanh":
        return tanh
    elif activation == "relu":
        return relu
    elif activation == "leaky_relu":
        return leaky_relu
    elif activation == "softmax":
        return softmax
    elif activation == "linear":
        return linear
    else:
        raise ValueError('Activation function not found')

def map_activaion_derivative(activation):
    if activation == "sigmoid":
        return sigmoid_derivative
    elif activation == "tanh":
        return tanh_derivative
    elif activation == "relu":
        return relu_derivative
    elif activation == "leaky_relu":
        return leaky_relu_derivative
    elif activation == "softmax":
        return softmax_derivative
    elif activation == "linear":
        return linear_derivative
    else:
        raise ValueError('Activation function not found')
