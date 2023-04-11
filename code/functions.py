import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Embedding

def load_csv(file_path):
    """
    loads a csv file as a numpy array
    args:
        file_path (str): path to the csv file
    returns:
        numpy.ndarray: The numpy array containing the data
    """
    data = np.genfromtxt(file_path, delimiter=',')
    return data

def split_data(data, test_size = 0.2, validate_size = 0.5, rand_seed = 123):
    """
    splits the data into training (80%), test (10%), and validation (10%) sets.
    args:
        data (numpy.ndarray): data to split
        test_size (float): proportion of data to use for the test/validate sets
        validate_size (float): proportion of test data to use for the validation set
        rand_seed (int): random seed to enable reproducibility
    returns:
        tuple of numpy.ndarrays: training, test, and validation sets
    """
    X = data[:, 1:]
    y = data[:, 0]
    
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size = test_size,
                                                      random_state = rand_seed,
                                                      stratify = y)
    
    X_test, X_valid, y_test, y_valid = train_test_split(X_rem, y_rem, 
                                                        test_size = validate_size,
                                                        random_state = rand_seed,
                                                        stratify = y_rem)
    return X_train, y_train, X_test, y_test, X_valid, y_valid

def train_LSTM(X_train, y_train, X_valid, y_valid):
    """
    build, compile, and fit an LSTM neural network classifier
    args:
        X_train, y_train, X_valid, y_valid: training and validation sets
    returns:
        trained LSTM model
    """
    # input for variable-length sequences of integers
    inputs = keras.Input(shape = (None, ), dtype = "int32")
    # embed each integer in a 28,128-dimensional vector
    x = layers.Embedding(X_train.shape[1], 128)(inputs)
    # add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences = True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    # Add a classifier
    outputs = layers.Dense(1, activation = "sigmoid")(x)
    model = keras.Model(inputs, outputs)

    # compile and train model
    model.compile("adam", "binary_crossentropy", metrics = ["accuracy"])
    model.fit(X_train, y_train, batch_size = 32, epochs = 2, 
              validation_data = (X_valid, y_valid))
    return model

def train_NN(X_train, y_train, X_valid, y_valid):
    """
    build, compile, and fit a neural network classifier.
    args:
        X_train, y_train, X_valid, y_valid: training and validation sets
    returns:
        trained NN model
    """
    model = Sequential()
    model.add(Dense(12, input_dim = 28, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(X_train, y_train, batch_size = 32, epochs = 10, 
              validation_data = (X_valid, y_valid))
    return model

# does not work - need < python 3.11.1
#def train_CNN(X_train):
#    """
#    build, compile, and fit a CNN autoencoder model on the data
#    args:
#        X_train: training set
#    returns:
#        trained CNN model
#    """
#    model = AutoEncoder(contamination = 0.05, hidden_neurons = [5, 5],
#                        epochs = 8, optimizer = 'Nadam', 
#                        loss = 'binary_crossentropy')
#    model.fit(X_train)
#    return model

def evaluate_model(model, X_test, y_test):
    """
    evaluate the performance of the trained model on a test set
    args:
        model: compiled and trained neural network model
        X_test, y_test: test sets
    returns: 
        precision, recall, and AUC scores
    """
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    area_under_curve = auc(recall, precision)
    return precision, recall, area_under_curve

def plot_precision_recall_curve(precision, recall, area_under_curve):
    """
    Plot the precision-recall curve for the trained model.
    args:
        precision: precision scores as a NumPy array
        recall: recall scores as a NumPy array
        area_under_curve: area under the precision-recall curve
    returns: 
        precision-recall curve plot
    """
    plt.plot(recall, precision, label='AUC = %0.2f' % area_under_curve)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc = 'lower left')
    plt.show()
