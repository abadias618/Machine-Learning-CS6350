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
    

    raw_train = read_data(train_file)
    raw_test = read_data(test_file)

    train = edit_labels(raw_train)
    test = edit_labels(raw_test)
    
    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)

    # stochastic sub-gradient descent -> ssgd
    if type_of_svm == "ssgd":
        
        C = [100/873, 500/873, 700/873]
        C_string = ["100/873", "500/873", "700/873"]
        schedule_params = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
        schedules = [1,2]
        for s in schedules:
            training = []
            testing = []
            for param in schedule_params:
                print("gamma",param)
                for i, c in enumerate(C):
                    svm = SVM(X_train, y_train, 100, c, schedule=s, gamma=param)
                    weights = svm.train()
                    train_predictions = svm.predict(X_train)
                    err, _ = average_prediction_error(train_predictions, y_train)
                    training.append(err)
                    print("training err",round(err,3), "with C:",C_string[i])
                    predictions = svm.predict(X_test)
                    err, _ = average_prediction_error(predictions, y_test)
                    testing.append(err)
                    print("testing err",err, "with C:",C_string[i])
            plt.plot(training, label="training")
            plt.plot(testing, label="testing")
            plt.legend()
            plt.savefig('withSchedule'+str(s) +'.png')
            plt.clf()

    if type_of_svm == "x":
        C = [100/873, 500/873, 700/873]
        C_string = ["100/873", "500/873", "700/873"]
        training = []
        testing = []
        for i, c in enumerate(C):
            svm = SVM(X_train, y_train, 100, c, schedule=1, gamma=0.05)
            weights = svm.train()
            print("weights",weights)
            train_predictions = svm.predict(X_train)
            err, _ = average_prediction_error(train_predictions, y_train)
            training.append(err)
            print("training err",round(err,3), "with C:",C_string[i])
            predictions = svm.predict(X_test)
            err, _ = average_prediction_error(predictions, y_test)
            testing.append(err)
            print("testing err",err, "with C:",C_string[i])

    if type_of_svm == "dual":
        C = [100/873, 500/873, 700/873]
        C_string = ["100/873", "500/873", "700/873"]
        training = []
        testing = []
        for i, c in enumerate(C):
            dual_svm = dual_SVM(X_train, y_train, 100, c)
            w, b = dual_svm.train()
            print("w, b", w, b)
            predictions = dual_svm.predict(X_test)
            err, _ = average_prediction_error(predictions, y_test)
            print("testing err",round(err,3), "with C:",C_string[i])