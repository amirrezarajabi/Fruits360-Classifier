from Activation import *
from Data import *

import numpy as np
import matplotlib.pyplot as plt
import time

class NN:
    def __init__(self, X, Y, input_size, output_size, epochs, learning_rate, batch_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.SIGMOID = "sigmoid"
        self.hidden_layers = [150, 60]
        self.X = X
        self.Y = Y
        self.W1, self.W2, self.W3 = None, None, None
        self.b1, self.b1, self.b3 = None, None, None
        self.a1, self.a2, self.a3 = None, None, None
        self.z1, self.z2, self.z3 = None, None, None
        self.accuracies = []
        self.costs = []
        self.time = 0
        self.initialize_weights()

    def initialize_weights(self):
        self.W1 = np.random.normal(0, 1, (self.hidden_layers[0], self.input_size))
        self.W2 = np.random.normal(0, 1, (self.hidden_layers[1], self.hidden_layers[0]))
        self.W3 = np.random.normal(0, 1, (self.output_size, self.hidden_layers[1]))
        self.b1 = np.zeros((self.hidden_layers[0], 1))
        self.b2 = np.zeros((self.hidden_layers[1], 1))
        self.b3 = np.zeros((self.output_size, 1))

    def shuffle_data(self):
        s = np.random.permutation(self.X.shape[1])
        self.X = self.X.T[s].T
        self.Y = self.Y.T[s].T
    
    def forward_propagation(self, x):
        self.z1 = self.W1 @ x + self.b1
        self.a1 = map_activaion(self.SIGMOID)(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = map_activaion(self.SIGMOID)(self.z2)
        self.z3 = self.W3 @ self.a2 + self.b3
        self.a3 = map_activaion(self.activation)(self.z3)
    
    def backward_propagation(self, x, y, g_w1, g_w2, g_w3, g_b1, g_b2, g_b3):
        self.forward_propagation(x)
        for j in range(self.W3.shape[0]):
            for k in range(self.W3.shape[1]):
                g_w3[j, k] += 2 * (self.a3[j, 0] - y[j, 0]) * map_activaion_derivative(self.activation)(self.z3[j, 0]) * self.a2[k, 0]
            g_b3[j, 0] += 2 * (self.a3[j, 0] - y[j, 0]) * map_activaion_derivative(self.activation)(self.z3[j, 0])
        
        
        da2 = np.zeros((self.a2.shape[0], 1))
        for k in range(self.a2.shape[0]):
            for j in range(self.a3.shape[0]):
                da2[k, 0] += self.W3[j, k] * 2 * (self.a3[j, 0] - y[j, 0]) * map_activaion_derivative(self.activation)(self.z3[j, 0])
        
        for k in range(self.W2.shape[0]):
            for m in range(self.W2.shape[1]):
                g_w2[k, m] += da2[k, 0] * map_activaion_derivative(self.SIGMOID)(self.z2[k, 0]) * self.a1[m, 0]
            g_b2[k, 0] += da2[k, 0] * map_activaion_derivative(self.SIGMOID)(self.z2[k, 0])
        
        
        da1 = np.zeros((self.a1.shape[0], 1))
        for m in range(self.a1.shape[0]):
            for k in range(self.a2.shape[0]):
                da1[m, 0] += self.W2[k, m] * da2[k, 0] * map_activaion_derivative(self.SIGMOID)(self.z2[k, 0])
        
        for m in range(self.W1.shape[0]):
            for v in range(self.W1.shape[1]):
                g_w1[m, v] += da1[m, 0] * map_activaion_derivative(self.SIGMOID)(self.z1[m, 0]) * x[v, 0]
            g_b1[m, 0] += da1[m, 0] * map_activaion_derivative(self.SIGMOID)(self.z1[m, 0])
        
    def accuracy_cost(self):
        accuracy = 0
        cost = 0
        for i in range(self.X.shape[1]):
            self.forward_propagation(self.X[:, i].reshape(self.input_size, 1))
            if np.argmax(self.a3) == np.argmax(self.Y[:, i].reshape(self.output_size, 1)):
                accuracy += 1
            cost += np.sum(np.square(self.a3 - self.Y[:, i].reshape(self.output_size, 1)))
        return accuracy / self.X.shape[1], cost / self.X.shape[1]

    def update_weights(self, g_w1, g_w2, g_w3, g_b1, g_b2, g_b3, b_s):
        self.W1 -= self.learning_rate * g_w1 / b_s
        self.W2 -= self.learning_rate * g_w2 / b_s
        self.W3 -= self.learning_rate * g_w3 / b_s
        self.b1 -= self.learning_rate * g_b1 / b_s
        self.b2 -= self.learning_rate * g_b2 / b_s
        self.b3 -= self.learning_rate * g_b3 / b_s
    
    def train(self):
        tic = time.time()
        for i in range(self.epochs):
            self.shuffle_data()
            for bc in range(self.X.shape[1] // self.batch_size):
                end = min(bc * self.batch_size + self.batch_size, self.X.shape[1])
                B_X = self.X[:, bc * self.batch_size:end]
                B_Y = self.Y[:, bc * self.batch_size:end]
                g_w1, g_w2, g_w3, g_b1, g_b2, g_b3 = np.zeros(self.W1.shape), np.zeros(self.W2.shape), np.zeros(self.W3.shape), np.zeros(self.b1.shape), np.zeros(self.b2.shape), np.zeros(self.b3.shape)
                for j in range(B_X.shape[1]):
                    self.backward_propagation(B_X[:, j].reshape(self.input_size, 1), B_Y[:, j].reshape(self.output_size, 1), g_w1, g_w2, g_w3, g_b1, g_b2, g_b3)
                self.update_weights(g_w1, g_w2, g_w3, g_b1, g_b2, g_b3, B_X.shape[1])
            accuracy, cost = self.accuracy_cost()
            self.accuracies.append(accuracy * 100)
            self.costs.append(cost)
        toc = time.time()
        self.time = (toc - tic) * 1000
    
    def show_result(self):
        print('Time:', self.time, 'ms')
        print('Accuracy:', f'{"{:.3f}".format(self.accuracies[-1])}', '%')
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(self.accuracies)
        ax1.set_title('Accuracy')
        ax2.plot(self.costs)
        ax2.set_title('Cost')
        plt.show()
    
    def first_assign_prediction(self):
        self.initialize_weights()
        return self.accuracy_cost()[0] * 100

X_train, Y_train, X_test, Y_test = preprocess()
X_t = X_train[:, :200]
Y_t = Y_train[:, :200]

nn = NN(X_t, Y_t, 102, 4, 5, 1, 10, "sigmoid")
acc = 0
N = 100
for i in range(N):
    acc += nn.first_assign_prediction()
print('Accuracy:', acc / N, '%', f"in {N} times for first time")
print("#####################################\n")
nn.train()
nn.show_result()
# nn.X = X_train
# nn.Y = Y_train
# print(nn.accuracy_cost()[0] * 100)