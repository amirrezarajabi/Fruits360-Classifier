from Loading_Datasets import load_dataset
import numpy as np

def preprocess():
    data = load_dataset()
    train = data[0]
    test = data[1]
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in train:
        X_train.append(i[0])
        y_train.append(i[1])
    for i in test:
        X_test.append(i[0])
        y_test.append(i[1])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    n_x = X_train.shape[1]
    S = X_train.shape[0]
    n_y = y_train.shape[1]
    S_test = X_test.shape[0]
    X_train = X_train.reshape(S, n_x)
    X_train = X_train.T
    y_train = y_train.reshape(S, n_y)
    y_train = y_train.T
    X_test = X_test.reshape(S_test, n_x)
    X_test = X_test.T
    y_test = y_test.reshape(S_test, n_y)
    y_test = y_test.T
    return X_train, y_train, X_test, y_test
