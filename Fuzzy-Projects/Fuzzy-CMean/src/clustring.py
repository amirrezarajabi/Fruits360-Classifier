import numpy as np

class Clustering:

    def __init__(self, c, m, data):
        self.C = c
        self.m = m
        self.data = data
        self.V = None
        self.U = None
    
    def initial_centeroid(self):
        self.V = np.zeros((self.C, self.data.shape[1]))
        min_ = np.min(self.data, 0)
        max_ = np.max(self.data, 0)
        for j in range(self.C):
            self.V[j, :] = self.generate_random_centroid(min_, max_)
    
    def generate_random_centroid(self, min_, max_):
        return np.random.uniform(low=min_, high=max_)
        


    def calculate_J(self):
        cost = 0
        N = self.data.shape[0]
        for j in range(N):
            for i in range(self.C):
                cost += np.power(self.U[i, j], self.m) * np.power(np.linalg.norm(self.data[j] - self.V[i]), 2)
        return cost

    def calculate_U(self):
        N = self.data.shape[0]
        self.U = np.zeros((self.C, N))
        for i in range(self.C):
            for k in range(N):
                tmp = 0
                for c in range(self.C):
                    tmp += (np.linalg.norm(self.data[k] - self.V[i]) / np.linalg.norm(self.data[k] - self.V[c])) ** (2 / (self.m - 1))
                self.U[i, k] = 1 / tmp
    
    def update_V(self):
        u_m = np.power(self.U, self.m)
        numerator = np.dot(u_m, self.data)
        denominator = np.sum(u_m, axis=1).reshape(-1, 1)
        self.V = numerator / denominator
    
    def run(self, iterations):
        self.initial_centeroid()
        for i in range(iterations):
            self.calculate_U()
            self.update_V()
        return self.calculate_J()