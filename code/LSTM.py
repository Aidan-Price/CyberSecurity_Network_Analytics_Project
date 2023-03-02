### LSTM neural network for cybersecurity network analytics
### Mike McCormick | University of Colorado Boulder

### import libraries
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score, classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC

# references and inspiration
# https://keras.io/examples/nlp/bidirectional_lstm_imdb/
# CU HIN - https://github.com/elgood/CU_HIN/tree/main/src

# create LSTM model
def LSTM_model(input_size):
    # variable length inputs
    inputs = keras.Input(shape = (None, ), dtype = "int32")    
    # embedding layer (input_size, 128)
    x = layers.Embedding(input_size, 128)(inputs)
    # add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences = True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)    
    # add classifier
    outputs = layers.Dense(1, activation = "sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model

# calculate metrics from test data
def metrics(y_test, predictions):
    print(classification_report(y_test, predictions))
    precision, recall, fscore = score(y_test, predictions, average = 'binary')
    auc_roc = roc_auc_score(y_test, predictions)
    
    return precision, recall, fscore, auc_roc
    
def main():
    # process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', type=str, required=True,
                      help='File with the features and labels; labels must be in 1st column')
    FLAGS = parser.parse_args()
    
    # open file
    with open(FLAGS.inputfile, "r") as infile:
        data = np.loadtxt(infile, delimiter=",")
        scenario_name = FLAGS.inputfile.replace('.txt', '')   
    
        # split into independent (X) and dependent (y) variables
        X = data[:, 1:]
        y = data[:, 0]
              
        # split into training (80%), validation (10%), and test (10%) sets
        rand_seed = 123
        X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size = 0.8, 
                                                          random_state = rand_seed,
                                                          stratify = y)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, 
                                                            train_size = 0.5,
                                                            random_state = rand_seed,
                                                            stratify = y_rem)
    
        # build and compile model with training and validation sets
        model = LSTM_model(X.shape[1])
        model.compile("adam", "binary_crossentropy", metrics = ["accuracy"])
        model.fit(X_train, y_train, batch_size = 32, epochs = 2, 
                  validation_data = (X_valid, y_valid))           
        
        # make predictions with model
        predictions = (model.predict(X_test) > 0.5).astype(int)
        
        # calculate metrics
        precision, recall, fscore, auc_roc = metrics(y_test, predictions)
        print('scenario_name', scenario_name)
        print('precision', precision)
        print('ecall', recall)
        print('fscore', fscore)
        print('auc_roc', auc_roc)
        
main()