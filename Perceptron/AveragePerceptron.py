import random

class AveragePerceptron():
    def __init__(self, X, y, epochs = 1, bias = 0, learning_rate = 1e-5):
        self.X = X
        self.y = y
        self.epochs = epochs
        self.weights = self.initialize_weights(len(self.X[0]))
        self.bias = bias
        self.learning_rate = learning_rate
        self.w_and_C_tuple_list = []
        self.a = self.initialize_weights(len(self.X[0])) 

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
        C = 0
        # fold bias into weights
        self.weights = self.weights + [self.bias]
        # fold bias into a
        self.a = self.a + [self.bias]

        for _ in range(self.epochs):
            X, y = self.shuffle_data(self.X, self.y)
            for i, row in enumerate(X):
                # fold bias into vector
                b_row = row + [self.bias]
                prediction = self.predict_single(b_row, self.weights)
                if y[i] != prediction:
                    self.weights = self.update(y[i], b_row, self.weights)

                self.a = [a + b for a,b in zip(self.a, self.weights)]
                
        return self.a, self.weights

    def update(self, y, X_i, weights):
        updated = []

        for i in range(len(weights)):
            updated.append(weights[i] + self.learning_rate * y * X_i[i])

        return updated

    def dot_product(self, a,b):
        result = 0
        for x, y in zip(a,b):
            result += (x * y)
        return result

    def sgn(self, x):
            if x <= 0:
                return -1
            else:
                return 1

    def predict_single(self, row, weights):
        product = self.dot_product(row, weights)

        return self.sgn(product)

    def predict(self, X):
        
        predictions = []
        for row in X:
            b_row = row + [self.bias]
            
            dot = self.dot_product(b_row, self.a)
            predictions.append(self.sgn(dot))

        return predictions


