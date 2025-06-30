# Description: This file is a solution to the digit recognizer Kaggle competition. The goal is to convert the solution to run using parallel processing on RIS.
# Author: Hayden Schroeder

# TODO: Convert the solution to run using parallel processing on RIS.
# TODO: Use pytorch instead of keras.

import pandas as pd
import numpy as np
from keras.utils import to_categorical


def load_data():
    """
    Load the training and test datasets.
    Returns:
        x_train: Training data features.
        y_train: Training data labels.
        x_test: Test data features.
    """

    # Load datasets
    train = pd.read_csv('train.csv')
    x_test = pd.read_csv('test.csv')

    x_train = train.drop(columns=['label'])
    y_train = train['label']

    # Reshape to 28x28 images with 1 channel (grayscale) for CNN input
    x_train = x_train.values.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.values.reshape((x_test.shape[0], 28, 28, 1))

    return x_train, y_train, x_test


seed = 33
np.random.seed(seed)


def linear_model():
    from keras.models import Sequential
    from keras.layers import Lambda, Dense, Flatten, Input, Dropout
    from keras.callbacks import EarlyStopping
    from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Lambda(standardize))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))


x_train, y_train, x_test = load_data()

# Feature standardization
mean = x_train.mean().astype(np.float32)
std = x_train.std().astype(np.float32)


def standardize(x):
    return (x - mean) / std


# One hot encoding of labels
y_train = to_categorical(y_train, num_classes=10)

linear_model()
