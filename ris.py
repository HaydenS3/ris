# Description: This file is a solution to the digit recognizer Kaggle competition. The goal is to convert the solution to run using parallel processing on RIS.
# Author: Hayden Schroeder

# TODO: Convert the solution to run using parallel processing on RIS.
# TODO: Use pytorch instead of keras.
# TODO: Add image augmentation to the model

import pandas as pd
import numpy as np
from keras.utils import to_categorical
import datetime
from keras.models import Sequential
from keras.layers import Lambda, Dense, Flatten, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras.src.legacy.preprocessing.image import ImageDataGenerator  # ImageDataGenerator has been deprecated in keras
from sklearn.model_selection import train_test_split

BATCH_SIZE = 64


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


def generate_batches(x, y, batch_size=64):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    generator = ImageDataGenerator()
    batches = generator.flow(x_train, y_train, batch_size=batch_size)
    test_batches = generator.flow(x_test, y_test, batch_size=batch_size)
    return batches, test_batches


def linear_model(x_train, y_train):
    """ Train a simple linear model on the MNIST dataset."""
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Lambda(standardize))
    model.add(Flatten())    # Converts to 2D tensor for dense layers
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    batches, test_batches = generate_batches(x_train, y_train, BATCH_SIZE)

    model.fit(batches, epochs=4, validation_data=test_batches,
              steps_per_epoch=batches.n, validation_steps=test_batches.n)

    # Save the model
    model.save(f'linear-model-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.keras')


def fully_connected_model(x_train, y_train):
    """ Train a fully connected model on the MNIST dataset."""
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Lambda(standardize))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    batches, test_batches = generate_batches(x_train, y_train, BATCH_SIZE)

    model.fit(batches, epochs=4, validation_data=test_batches,
              steps_per_epoch=batches.n, validation_steps=test_batches.n)

    # Save the model
    model.save(f'fully-connected-model-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.keras')


def cnn_model(x_train, y_train):
    """ Train a CNN model on the MNIST dataset."""
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Lambda(standardize))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(523, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    batches, test_batches = generate_batches(x_train, y_train, BATCH_SIZE)

    model.fit(batches, epochs=4, validation_data=test_batches,
              steps_per_epoch=batches.n, validation_steps=test_batches.n)

    # Save the model
    model.save(f'cnn-model-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.keras')


x_train, y_train, x_test = load_data()

# Feature standardization
mean = x_train.mean().astype(np.float32)
std = x_train.std().astype(np.float32)


def standardize(x):
    return (x - mean) / std


# One hot encoding of labels
y_train = to_categorical(y_train, num_classes=10)

# linear_model(x_train, y_train)
# fully_connected_model(x_train, y_train)
cnn_model(x_train, y_train)
