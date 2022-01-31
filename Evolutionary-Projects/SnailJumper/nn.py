import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.W = []
        self.B = []

        for i in range(1, len(layer_sizes)):
            self.W.append(np.random.normal(0, 1, (layer_sizes[i], layer_sizes[i - 1])))
            self.B.append(np.zeros((layer_sizes[i], 1)))

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        return self.map_activaion("sigmoid")(x)

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        A = None
        for i in range(len(self.W)):
            if i == 0:
                A = self.activation(np.dot(self.W[i], x) + self.B[i])
            else:
                A = self.activation(np.dot(self.W[i], A) + self.B[i])
        return A

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def leaky_relu(self, x):
        return np.maximum(0.01 * x, x)

    def softmax(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)

    def linear(self, x):
        return x

    def map_activaion(self, activation):
        if activation == "sigmoid":
            return self.sigmoid
        elif activation == "tanh":
            return self.tanh
        elif activation == "relu":
            return self.relu
        elif activation == "leaky_relu":
            return self.leaky_relu
        elif activation == "softmax":
            return self.softmax
        elif activation == "linear":
            return self.linear
        else:
            raise ValueError('Activation function not found')