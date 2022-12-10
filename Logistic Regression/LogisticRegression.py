import numpy as np
from random import shuffle
from math import e

class LogisticRegression():
    def __init__(self, X, y, v, T=100, gamma=0.05, a=0.02, estimation_type="map"):
        self.X = X
        self.y = y
        self.v = v
        self.T = T
        self.gamma = gamma
        self.a = a
        self.N = len(X)
        self.estimation_type = estimation_type
        self.weights = self.initialize_weights(len(X[0]))


    def initialize_weights(self, row_length):
        #w = np.matrix(np.zeros(row_length)).T
        w = np.zeros(row_length)
        return w
    
    def schedule_1(self, t):
        return self.gamma / (1 + ((self.gamma/self.a) * t))

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
                linear_model = np.dot(row, self.weights)
                pred = self.sigmoid(linear_model)
                if self.estimation_type == "ml":
                    dw = 1 - pred
                elif self.estimation_type == "map":
                    dw = (1 / self.v) * np.dot(row, (pred - y[i]))
                
                self.weights -= self.schedule_1(t+1) * dw

    def sgn(self, x):
            if x <= 0:
                return -1
            else:
                return 1
    
    def sigmoid(self, x):
        return 1 / (1+ np.exp(-x))

    def predict(self, X):
        predictions = []
        for row in X:
            dot = np.dot(row, self.weights)
            predictions.append(self.sgn(dot))
        return predictions