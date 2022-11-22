import random
import numpy as np
from scipy.optimize import  minimize

class dual_SVM():
    def __init__(self, X, y, T, C, bias=0, learning_rate=1e-5):
        self.X = X
        self.y = y
        self.T = T
        self.C = C
        self.N = len(self.X)


    def train(self):
        a = np.zeros(self.N)
        k = np.ndarray([self.N, self.N])

        