import unittest
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
from code.functions import train_LSTM, train_NN, evaluate_model

class TestMyModule(unittest.TestCase):
                
    def test_train_LSTM(self):
        # Test that the function trains an LSTM model and returns a keras model object
        X_train = np.random.randint(0, 10, size=(100, 10))
        y_train = np.random.randint(0, 2, size=100)
        X_valid = np.random.randint(0, 10, size=(20, 10))
        y_valid = np.random.randint(0, 2, size=20)
        model = train_LSTM(X_train, y_train, X_valid, y_valid)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertIsInstance(model, keras.models.Model)
        
    def test_train_NN(self):
        # Test that the function trains a neural network model and returns a keras model object
        X_train = np.random.randint(0, 10, size=(100, 28))
        y_train = np.random.randint(0, 2, size=100)
        X_valid = np.random.randint(0, 10, size=(20, 28))
        y_valid = np.random.randint(0, 2, size=20)
        model = train_NN(X_train, y_train, X_valid, y_valid)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertIsInstance(model, keras.models.Model)
        
#   def test_train_CNN(self):
#        # Test that the function trains a CNN model and returns a pyod AutoEncoder object
#        X_train = np.random.rand(100, 10)
#        model = train_CNN(X_train)
#        self.assertTrue(hasattr(model, 'predict'))
#        self.assertIsInstance(model, AutoEncoder)
        
    def test_evaluate_model(self):
        # Test that the function returns precision, recall, and AUC scores
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(1, input_dim=10, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        X_test = np.random.randint(0, 10, size=(20, 10))
        y_test = np.random.randint(0, 2, size=20)
        precision, recall, auc_score = evaluate_model(model, X_test, y_test)
        self.assertIsInstance(precision, np.ndarray)
        self.assertIsInstance(recall, np.ndarray)
        self.assertIsInstance(auc_score, float)
    
if __name__ == '__main__':
    unittest.main()
