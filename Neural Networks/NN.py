from random import shuffle
from math import e
import numpy as np

class NeuralNet():
    def __init__(self, X, y, input, layer_1, layer_2, output_layer, T=10, gamma=0.05, a=0.02, zero_weights=False):
        self.X = X
        self.y = y
        self.weights = self.initialize_weights(input, layer_1, layer_2, output_layer) if zero_weights == False else self.initialize_zero_weights(input, layer_1, layer_2, output_layer)
        self.input = input
        self.layer_1 = np.ones(layer_1)
        self.layer_2 = np.ones(layer_2)
        self.T = T
        self.output_layer = output_layer
        self.gamma = gamma
        self.a = a
        

    def schedule_1(self, t):
        return self.gamma / (1 + ((self.gamma/self.a) * t))
    
    def initialize_weights(self, input, layer_1, layer_2, output_layer):
        w = []
        w.append(np.random.normal(size = input * (layer_1 - 1)))
        w.append(np.random.normal(size = layer_1 * (layer_2 - 1)))
        w.append(np.random.normal(size = layer_2 * output_layer))
        return w

    def initialize_zero_weights(self, input, layer_1, layer_2, output_layer):
        w = []
        w.append(np.zeros(input * (layer_1 - 1)))
        w.append(np.zeros(layer_1 * (layer_2 - 1)))
        w.append(np.zeros(layer_2 * output_layer))
        return w

    def shuffle_data(self, X, y):
        merged_X_y = [[X[i] , y[i]] for i in range(len(X))]
        shuffle(merged_X_y)
        s_X = [merged_X_y[i][0] for i in range(len(merged_X_y))]
        s_y = [merged_X_y[i][1] for i in range(len(merged_X_y))]
        return s_X, s_y

    def train(self):
        for t in range(self.T):
            X, y = self.shuffle_data(self.X, self.y)
            for i, row in enumerate(X):
                self.backward([1] + row, y[i], self.schedule_1(t+1))
    
    def sgn(self, x):
            if x <= 0:
                return -1
            else:
                return 1
    
    def sigmoid(self, x):
        return 1 / (1+ e ** (-x))

    def forward(self, row):
        for i in range(1,len(self.layer_1)):
            sum = 0
            for j in range(len(row)):
                if j+(i-1)*len(row) < len (self.weights[0]): ####
                    sum += row[j] * self.weights[0][j+(i-1)*len(row)]
            self.layer_1[i] = self.sigmoid(sum)

        for i in range(1,len(self.layer_2)):
            sum = 0
            for j in range(len(self.layer_1)):
                sum += self.layer_1[j] * self.weights[1][j+(i-1)*len(self.layer_1)]
                    
            self.layer_2[i] = self.sigmoid(sum)

        y = 0
        for i in range(len(self.layer_2)):
            y += self.layer_2[i] * self.weights[2][i]
        return y

    def backward(self, row, y, learning_rate):
        pred = self.forward(row)

        for i in range(len(self.weights[2])):
            self.weights[2][i] -= learning_rate * (pred - y) * self.layer_2[i%len(self.layer_2)]
        for i in range(len(self.weights[1])):
            self.weights[1][i] -= learning_rate * (pred - y) * self.weights[2][int(i/len(self.layer_1)) + 1] * self.layer_1[i%len(self.layer_1)]
        for i in range(len(self.weights[0])):
            sum = 0
            for j in range(len(self.layer_2)):
                sum += self.weights[2][j] * self.weights[1][(j-1) * len(self.layer_1) + int(i/len(self.layer_1)) + 1]
            self.weights[0] -= learning_rate * (pred - y) * sum * row[i%len(row)]

    def predict(self, X):
        predictions = []
        for row in X:
            predictions.append(self.sgn(self.forward(row)))
        return predictions