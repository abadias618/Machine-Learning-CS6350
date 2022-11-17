import random

class SVM():
    def __init__(self, X, y, T, bias=0, learning_rate=1e-5):
        self.X = X
        self.y = y
        self.T = T
        self.weights = self.initialize_weights(len(self.X[0]))
        self.bias = bias
        self.learning_rate = learning_rate
    
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
            for i, row in zip(enumerate(X),y):
                # fold bias into vector
                b_row = row + [self.bias]
                x = self.scalar_by_vec(y[i], self.dot_product(self.weights, b_row)) # y*w*x
                if  x <= 1:
                    self.weights = self.update(y[i], b_row, self.weights)
                else:
                    
                prediction = self.predict_single(b_row, self.weights)
                if y[i] != prediction:
                    #update weights
                    self.weights = self.update(y[i], b_row, self.weights)
                 
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
    def update(self, y, X_i, weights):
        updated = []

        for i in range(len(weights)):
            updated.append(weights[i] + self.learning_rate * y * X_i[i])

        return updated