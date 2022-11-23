import random

class SVM():
    def __init__(self, X, y, T, C, bias=0, learning_rate=1e-5):
        self.X = X
        self.y = y
        self.T = T
        self.C = C
        self.N = len(self.X)
        self.initial_weights = self.initialize_weights(len(self.X[0]))
        self.weights = self.initialize_weights(len(self.X[0]))
        self.bias = bias
        self.learning_rate = learning_rate

    def schedule_1(self, gamma):
        return
    
    def schedule_1(self, gamma):
        return

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
        # fold bias into weights
        self.weights = self.weights + [self.bias]

        for _ in range(self.T):
            X, y = self.shuffle_data(self.X, self.y)
            for i, row in enumerate(X):
                # fold bias into vector
                b_row = row + [self.bias]
                x = y[i] * self.dot_product(self.weights, b_row) # y*w*x
                if  x <= 1:
                    self.weights = self.update(y[i], b_row,
                                                self.weights, self.initial_weights,
                                                self.learning_rate, self.C,
                                                self.N)
                else:
                    self.initial_weights = self.update_initial_weights(self.initial_weights, self.learning_rate)
                           
        return self.weights

    def scalar_by_vec(self, s,v):
        vec = []
        for x in v:
            vec.append(s*x)
        return vec

    def dot_product(self, a,b):
        result = 0
        for x, y in zip(a,b):
            result += (x * y)
        return result

    def update(self, y_i, X_i, weights, initial_weights, learning_rate, C, N):
        updated = []
        a = self.scalar_by_vec(learning_rate, initial_weights + [0]) # fold bias into W_0
        b = self.scalar_by_vec((learning_rate * C * N * y_i ), X_i)
        for i in range(len(weights)):
            updated.append(weights[i] - a[i] + b[i])

        return updated
    
    def update_initial_weights(self, initial_weights, learning_rate):
        s = 1 - learning_rate
        return self.scalar_by_vec(s, initial_weights)
    
    def sgn(self, x):
            if x <= 0:
                return -1
            else:
                return 1

    def predict(self, X):
        predictions = []
        for row in X:
            b_row = row + [self.bias]
            dot = self.dot_product(b_row, self.weights)
            predictions.append(self.sgn(dot))

        return predictions