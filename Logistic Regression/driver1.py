import sys
import csv
from LogisticRegression import *

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
    estimation_type = sys.argv[3]
    

    raw_train = read_data(train_file)
    raw_test = read_data(test_file)

    train = edit_labels(raw_train)
    test = edit_labels(raw_test)
    
    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)

    if estimation_type == "map":
        V = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
        for v in V:
            lr = LogisticRegression(X_train, y_train, v)
            lr.train()
            train_preds = lr.predict(X_train)
            print("training error:",round(average_prediction_error(train_preds,y_train)[0],3))
            preds = lr.predict(X_test)
            print("testing error:",round(average_prediction_error(preds,y_test)[0],3))
    elif estimation_type == "ml":
        V = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
        for v in V:
            lr = LogisticRegression(X_train, y_train, v, estimation_type="ml")
            lr.train()
            train_preds = lr.predict(X_train)
            print("training error:",round(average_prediction_error(train_preds,y_train)[0],3))
            preds = lr.predict(X_test)
            print("testing error:",round(average_prediction_error(preds,y_test)[0],3))