import sys
import csv
from NN import *
import torch
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
    type_of_nn = sys.argv[3]
    

    raw_train = read_data(train_file)
    raw_test = read_data(test_file)

    train = edit_labels(raw_train)
    test = edit_labels(raw_test)
    
    X_train, y_train = split_features_labels(train)
    X_test, y_test = split_features_labels(test)

    if type_of_nn == "gaussian":
        widths = [5, 10, 25, 50, 100]
        epochs = [100,75,50,10,10]
        for w, e in zip(widths,epochs):
            nn = NeuralNet(X_train, y_train, input=3, layer_1=w, layer_2=w, output_layer=1, T=e)
            nn.train()
            train_preds = nn.predict(X_train)
            print("training error for width:",w,"error:",round(average_prediction_error(train_preds,y_train)[0],3))
            preds = nn.predict(X_test)
            print("testing error",w,"error:",round(average_prediction_error(preds,y_test)[0],3))
        
    elif type_of_nn == "zeros":
        widths = [5, 10, 25, 50, 100]
        epochs = [100,75,50,10,10]
        for w, e in zip(widths,epochs):
            nn = NeuralNet(X_train, y_train, input=3, layer_1=w, layer_2=w, output_layer=1, T=e, zero_weights=True)
            nn.train()
            train_preds = nn.predict(X_train)
            print("training error for width:",w,"error:",round(average_prediction_error(train_preds,y_train)[0],3))
            preds = nn.predict(X_test)
            print("testing error",w,"error:",round(average_prediction_error(preds,y_test)[0],3))
    
    elif type_of_nn == "ReLU":
        #make datasets Tensor datasets
        activation = "ReLU"
        T = 100
        for i in range(len(X_train)):
            X_train[i] = torch.Tensor(X_train[i])
            y_train[i] = torch.Tensor([y_train[i]])
        for i in range(len(X_test)):
            X_test[i] = torch.Tensor(X_test[i])
            y_test[i] = torch.Tensor([y_test[i]])

        for depth in [3,5,9]:
            for width in [5,10,25,50,100]:
                make_activation = (lambda: torch.nn.ReLU())
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, width),
                    *[layer for _ in range(depth - 2) for layer in [make_activation(), torch.nn.Linear(width, width)]],
                    make_activation(),
                    torch.nn.Linear(width, 1)
                )

                def init_weights(m):
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.kaiming_normal_(m.weight)
                        m.bias.data.fill_(0.01)
                model.apply(init_weights)

                loss_fn = torch.nn.MSELoss(reduction="sum")
                optimizer = torch.optim.Adam(model.parameters())

                for _ in range(T):
                    for x, y in zip(X_train, y_train):
                        y_pred = model(x)
                        loss = loss_fn(y_pred, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                def predict(X, y):
                    predictions = []
                    for x, y_ in zip(X, y):
                        y_pred = -1 if model(x)[0] < 0 else 1
                        predictions.append(y_pred)
                    return predictions
                    
                print("depth:", depth, "width:",width)
                print("Training error:",round(average_prediction_error(y_train, predict(X_train, y_train))[0], 3))
                print("Testing error:",round(average_prediction_error(y_test, predict(X_test, y_test))[0], 3))
                print()

    elif type_of_nn == "Tanh":
        activation = "tanh"
        T = 100
        #make datasets Tensor datasets
        for i in range(len(X_train)):
            X_train[i] = torch.Tensor(X_train[i])
            y_train[i] = torch.Tensor([y_train[i]])
        for i in range(len(X_test)):
            X_test[i] = torch.Tensor(X_test[i])
            y_test[i] = torch.Tensor([y_test[i]])

        for depth in [3,5,9]:
            for width in [5,10,25,50,100]:
                make_activation = (lambda: torch.nn.Tanh())
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, width),
                    *[layer for _ in range(depth - 2) for layer in [make_activation(), torch.nn.Linear(width, width)]],
                    make_activation(),
                    torch.nn.Linear(width, 1)
                )

                def init_weights(m):
                    if isinstance(m, torch.nn.Linear):    
                        torch.nn.init.xavier_uniform_(m.weight)
                        m.bias.data.fill_(0.01)

                model.apply(init_weights)

                loss_fn = torch.nn.MSELoss(reduction="sum")
                optimizer = torch.optim.Adam(model.parameters())

                for _ in range(T):
                    for x, y in zip(X_train, y_train):
                        y_pred = model(x)
                        loss = loss_fn(y_pred, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                def predict(X, y):
                    predictions = []
                    for x, y_ in zip(X, y):
                        y_pred = -1 if model(x)[0] < 0 else 1
                        predictions.append(y_pred)
                    return predictions

                print("depth:", depth, "width:",width)
                print("Training error:",round(average_prediction_error(y_train, predict(X_train, y_train))[0], 3))
                print("Testing error:",round(average_prediction_error(y_test, predict(X_test, y_test))[0], 3))
                print()