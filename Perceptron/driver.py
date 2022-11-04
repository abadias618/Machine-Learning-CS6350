import sys
import csv

from Perceptron import *
from VotedPerceptron import *
from AveragePerceptron import *

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
    type_of_perceptron = sys.argv[3]

    raw_train = read_data(train_file)
    raw_test = read_data(test_file)

    train = edit_labels(raw_train)
    test = edit_labels(raw_test)
    
    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)
    
    if type_of_perceptron == 'standard':
        print('\n\nSTANDARD PERCEPTRON')
        p = Perceptron(X_train, y_train, epochs=10)

        weights_v = p.train()
        print('learned weight vector:',weights_v,'last element is the folded bias')
        predictions = p.predict(X_test)
        err, _ = average_prediction_error(predictions, y_test)
        print('Average Prediction ERROR %',err)

    elif type_of_perceptron == 'voted':
        print('\n\nVOTED PERCEPTRON')
        p = VotedPerceptron(X_train, y_train, epochs=10)

        w_and_c = p.train()
        print('distinct weights vectors and lifespan of vectors:',w_and_c)

        training_predictions = p.predict(X_train)
        _, t_correct = average_prediction_error(training_predictions, y_train)
        print('Correctly predicted training examples', t_correct)

        v_predictions = p.predict(X_test)
        err, correct = average_prediction_error(v_predictions, y_test)
        print('Average Prediction ERROR %',err)

    elif type_of_perceptron == 'average':
        print('\n\nAVERAGE PERCEPTRON')
        p = AveragePerceptron(X_train, y_train, epochs=10)

        a, weights = p.train()
        print('a vector',a, 'last element is the folded bias')
        print('learned weights vector', weights,'last element is the folded bias')
        a_predictions = p.predict(X_test)
        err, _ = average_prediction_error(a_predictions, y_test)
        print('Average Prediction ERROR %',err)