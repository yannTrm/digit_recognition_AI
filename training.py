# -*- coding: utf-8 -*-
# made by Yann Terrom, ESME Sudria x Paris Sclay student in AI

# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

import LeNet4
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Function
def load_process_data_MNIST():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0   
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train, x_test, y_test)
    
def load_process_data(path_data_train, path_data_test):     
    train_data = pd.read_csv(path_data_train, header = None)
    test_data = pd.read_csv(path_data_test, header = None)
    
    x_train, y_train = train_data.loc[:, 1:], train_data.loc[:, 0]
    y_train = keras.utils.to_categorical(y_train, 10)
    x_train = x_train.values.reshape(-1, 28, 28)
     
    x_test, y_test = test_data.loc[:, 1:], test_data.loc[:, 0]
    y_test = keras.utils.to_categorical(y_test, 10)
    x_test = x_test.values.reshape(-1, 28, 28)
    return (x_train, y_train, x_test, y_test)

def model_LeNet4():
    return keras.Sequential([
            layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
            layers.Conv2D(filters=20, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=50, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(units=500, activation="relu"),
            layers.Dense(units=10, activation="softmax")])


def boosted_training(model, x_train, y_train, n_iter = 10):
    models = []
    weights = np.ones(len(x_train)) / len(x_train)
    for _ in range(n_iter):
        model.fit(x_train, y_train, batch_size=128, epochs=5,
                  verbose=1, sample_weight=weights)
        models.append(model)
        predictions = model.predict(x_train)
        incorrect_predictions = np.argmax(predictions, axis=1) != np.argmax(y_train, axis=1)
        error = np.mean(incorrect_predictions)
        if error == 0:
            alpha = 0
        else:
            error = np.clip(error, 1e-15, 1 - 1e-15)
            alpha = 0.5 * np.log((1.0 - error) / error)
        weights *= np.exp(alpha * incorrect_predictions)
        weights /= np.sum(weights)
    return models
    
def save_boosted_model(models, path_to_save = "./boosted_model/", model_name = "model"):
    for i, model in enumerate(models):
        model.save(f"{path_to_save}{model}_{i}.h5")
     
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------  
if __name__=="__main__" :
    
    # using MNIST data (LeNet4)
    (x_train, y_train, x_test, y_test) = load_process_data_MNIST()
    model = model_LeNet4()
    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)
    model.save("model.h5")
    
    # using your own data
    path_data_train = "path_to_your_data_train"
    path_data_test = "path_to_your_data_train"
    (x_train, y_train, x_test, y_test) = load_process_data(path_data_train, path_data_test)
    model_boosted = model_LeNet4()
    model_boosted.compile(loss="categorical_crossentropy", 
                  optimizer="adam", metrics=["accuracy"])
    models = boosted_training(model_boosted, x_train, y_train, n_iter=5)
    save_boosted_model(models)
    
#------------------------------------------------------------------------------