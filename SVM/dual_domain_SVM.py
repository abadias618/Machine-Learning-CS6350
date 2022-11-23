import random
import numpy as np
from scipy.optimize import  minimize, Bounds

class dual_SVM():
    def __init__(self, X, y, T, C, bias=0, learning_rate=1e-5):
        self.X = X
        self.y = y
        self.T = T
        self.C = C
        self.N = len(self.X)
        self.weights = self.initialize_weights(len(self.X[0]))
        self.alpha = None
        self.bias = bias

    def initialize_weights(self, length_of_row):
        weights = []
        for i in range(length_of_row):
            weights.append(0)
        return weights

    def constraints(self, alphas, y):
            constraints = np.sum(alphas*y)
            return constraints

    def dual_loss_function(self, alphas, xxyy):
        alphas = alphas.reshape(-1,1)
        aa = alphas.dot(alphas.T)
        return 0.5*np.sum(aa*xxyy) - np.sum(alphas)

    def train(self):
        X, y = np.asarray(self.X), np.asarray(self.y)
        yy = y.reshape(-1,1).dot(y.reshape(-1,1).T)
        xx = X.dot(X.T)
        xxyy = xx*yy
        c_dict = {"type":"eq", "fun": self.constraints, "args":[y]}
        constraints = c_dict
        g = self.C * np.ones((self.N, 1)) / 2
        boundaries = Bounds(0, self.C)
        a_star = minimize(self.dual_loss_function, x0 = g,
                            args = (xxyy), method = "SLSQP",
                            bounds=boundaries,
                            constraints=constraints,
                            options={"maxiter":10})
        self.alpha = a_star.x
        print(a_star.fun)
        print(a_star.success)
        print(a_star.x)
        print(a_star.status)
        return a_star
        
    def predict(self, X):
        X = np.asarray(X)
        p = np.asarray(self.weights).dot(X.T) + self.bias >= 0
        p = p * 2 - 1
        return p