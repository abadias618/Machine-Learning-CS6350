import random
import numpy as np

class NeuralNet():
    def __init__(self, input, layer_1, layer_2, output, T=100, gamma=0.05, a=0.02):
        self.input = input
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.T = T
        self.output = output
        self.gamma = gamma
        self.a = a
        self.weights = self.initialize_weights()

    def schedule_1(self, t):
        return self.gamma / (1 + ((self.gamma/self.a) * t))
    
    def initialize_weights(self, length_of_row):
        weights = []
        for i in range(length_of_row):
            weights.append(0)
        return weights
    
    def shuffle_data(self, X, y):
        merged_X_y = [[X[i] , y[i]] for i in range(len(X))]
        random.shuffle(merged_X_y)
        s_X = [merged_X_y[i][0] for i in range(len(merged_X_y))]
        s_y = [merged_X_y[i][1] for i in range(len(merged_X_y))]
        return s_X, s_y

    def train(self):
        for t in range(self.T):
            sum = 0
            err = 0
            X, y = self.shuffle_data(self.X, self.y)
            for i, row in enumerate(X):
                if self.sgn(self.output())
    
    def sgn(self, x):
            if x <= 0:
                return -1
            else:
                return 1
    
    def output(self, row):
        for i in range(1,self.layer_1):
            sum = 0
            for j in range(len(row)):
                sum = row[j] * self.weights[0][j+(i-1)*len(row)]