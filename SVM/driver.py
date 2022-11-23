import sys
import csv
import matplotlib.pyplot as plt
from ssgd_SVM import *
from dual_domain_SVM import *

def read_data(file_name):
    """Returns 2D list with int csv data"""
    data = []
    with open(file_name) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            data.append(list(map(float, row)))
    return data

def edit_labels(list_2d):
    """Edits label to be {-1,1} instead of {0,1}"""
    for row in list_2d:
        for i in range(len(row)):
            if i == len(row)-1:
                if row[i] == 0:
                    row[i] = -1
    return list_2d
    
def split_features_labels(list_2d):
    X = []
    y = []
    for row in list_2d:
        features = []
        for i in range(len(row)):
            if i == len(row)-1:
                y.append(row[i])
            else:
                features.append(row[i])
        X.append(features)

    return X, y

def average_prediction_error(preds, y):
    if len(y) != len(preds):
        raise RuntimeError('Cannot calculate error of different sized lists.',len(preds),len(y))
    correct = 0
    for a,b in zip(preds, y):
        if a == b:
            correct += 1
    return (len(preds) - correct) / len(preds), correct

if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    type_of_svm = sys.argv[3]
    #constant = sys.argv[4]
    #epochs = sys.argv[5]
    #learning_rate = sys.argv[6]
    
    # dual SVM
    # Gaussian Kernel
    # Kernel Perceptron

    raw_train = read_data(train_file)
    raw_test = read_data(test_file)

    train = edit_labels(raw_train)
    test = edit_labels(raw_test)
    
    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)

    # stochastic sub-gradient descent -> ssgd
    if type_of_svm == "ssgd":
        training = []
        testing = []
        C = [100/873, 500/873, 700/873]
        C_string = ["100/873", "500/873", "700/873"]
        schedule_params = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        #schedule_params = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
        for param in schedule_params:
            print("gamma",param)
            for i, c in enumerate(C):
                svm = SVM(X_train, y_train, 100, c, learning_rate=param)
                weights = svm.train()
                train_predictions = svm.predict(X_train)
                err, _ = average_prediction_error(train_predictions, y_train)
                training.append(err)
                print("training err",err, "with C:",C_string[i])
                predictions = svm.predict(X_test)
                err, _ = average_prediction_error(predictions, y_test)
                testing.append(err)
                print("testing err",err, "with C:",C_string[i])
        plt.plot(training, label="training")
        plt.plot(testing, label="testing")
        plt.legend()
        plt.savefig('comparison2.png')

    if type_of_svm == "dual":
        dual_svm = dual_SVM(X_train, y_train, 1, 100/873, learning_rate=0.5)
        weights = dual_svm.train()
        print(*weights)
        predictions = dual_svm.predict(X_test)
        print("predictions", predictions)
        #err, _ = average_prediction_error(predictions, y_test)
        #print("err",err)