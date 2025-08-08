# Description: This file is a solution to the digit recognizer Kaggle competition. The goal is to convert the solution to run using parallel processing on RIS with GPU acceleration.
# Author: Hayden Schroeder

import pandas as pd
import numpy as np
from keras.utils import to_categorical
import datetime
from keras.models import Sequential
from keras.layers import Lambda, Dense, Flatten, Input, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
import itertools
from mpi4py import MPI
from tqdm import tqdm
import os
import tensorflow as tf
import sys

# GPU Configuration
# Check if GPU is available and configure it
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Set GPU as the preferred device
        tf.config.experimental.set_visible_devices(gpus, 'GPU')

        # Allow soft placement - if GPU ops fail, fall back to CPU
        tf.config.set_soft_device_placement(True)

        print(f"GPU(s) available: {len(gpus)} GPU(s) detected")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU devices found. Exiting")
    sys.exit(1)

# Create a GPU strategy for distributed training if multiple GPUs are available
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
else:
    strategy = None

CNN_HYPERPARAMETERS = {
    # 'conv_filters': [(16, 16), (16, 32), (32, 32), (32, 64), (64, 64), (64, 128)],
    'conv_filters': [(32, 64)],
    'dense_units': [64, 128, 256, 512],
    # 'learning_rate': [0.001, 0.01, 0.1],
    'learning_rate': [0.001, 0.01],
    # 'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'dropout_rate': [0.3, 0.5],
    # 'validation_split': [0.1, 0.2, 0.3],
    'validation_split': [0.2],
    # 'batch_size': [32, 64, 128, 256],
    'batch_size': [64, 256],
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only
tf.get_logger().setLevel('ERROR')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

train_file = '/app/train.csv'
test_file = '/app/test.csv'
out_dir = '/app/store'


def load_data():
    """
    Load the training and test datasets.
    Returns:
        x_train: Training data features.
        y_train: Training data labels.
        x_test: Test data features.
    """

    # Load datasets
    train = pd.read_csv(train_file)
    x_test = pd.read_csv(test_file)

    x_train = train.drop(columns=['label'])
    y_train = train['label']

    # Reshape to 28x28 images with 1 channel (grayscale) for CNN input
    x_train = x_train.values.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.values.reshape((x_test.shape[0], 28, 28, 1))

    # One hot encoding of labels
    y_train = to_categorical(y_train, num_classes=10)

    return x_train, y_train, x_test


# Set a random seed for reproducibility
seed = 33
np.random.seed(seed)


def generate_batches(x, y, batch_size=64, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    # Use GPU-optimized data pipeline
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(train_dataset.cardinality()).batch(batch_size).repeat()
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Optimize GPU utilization

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(batch_size).repeat()
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)  # Optimize GPU utilization

    train_dataset.n = len(x_train) // batch_size
    test_dataset.n = len(x_test) // batch_size

    return train_dataset, test_dataset


def linear_model(x_train, y_train):
    """ Train a simple linear model on the MNIST dataset."""
    if strategy:
        with strategy.scope():
            model = Sequential()
            model.add(Input(shape=(28, 28, 1)))
            model.add(Lambda(standardize))
            model.add(Flatten())    # Converts to 2D tensor for dense layers
            model.add(Dense(10, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    else:
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            model = Sequential()
            model.add(Input(shape=(28, 28, 1)))
            model.add(Lambda(standardize))
            model.add(Flatten())    # Converts to 2D tensor for dense layers
            model.add(Dense(10, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    batches, test_batches = generate_batches(x_train, y_train, BATCH_SIZE)

    model.fit(batches, epochs=4, validation_data=test_batches,
              steps_per_epoch=batches.n, validation_steps=test_batches.n)

    model.save(f'linear-model-gpu-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.keras')


def fully_connected_model(x_train, y_train):
    """ Train a fully connected model on the MNIST dataset."""
    if strategy:
        with strategy.scope():
            model = Sequential()
            model.add(Input(shape=(28, 28, 1)))
            model.add(Lambda(standardize))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    else:
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
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

    model.save(f'fully-connected-model-gpu-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.keras')


def cnn_model(x_train, y_train, hyperparameters):
    """ Train a CNN model on the MNIST dataset with GPU acceleration."""
    if strategy:
        with strategy.scope():
            model = Sequential()
            model.add(Input(shape=(28, 28, 1)))
            model.add(Lambda(standardize))
            model.add(Convolution2D(hyperparameters['conv_filters'][0], (3, 3), activation='relu'))
            model.add(Convolution2D(hyperparameters['conv_filters'][0], (3, 3), activation='relu'))
            model.add(MaxPooling2D())
            model.add(Dropout(hyperparameters['dropout_rate']))
            model.add(Convolution2D(hyperparameters['conv_filters'][1], (3, 3), activation='relu'))
            model.add(Convolution2D(hyperparameters['conv_filters'][1], (3, 3), activation='relu'))
            model.add(MaxPooling2D())
            model.add(Dropout(hyperparameters['dropout_rate']))
            model.add(Flatten())
            model.add(Dense(hyperparameters['dense_units'], activation='relu'))
            model.add(Dropout(hyperparameters['dropout_rate']))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer=Adam(
                learning_rate=hyperparameters['learning_rate']), metrics=['accuracy'])
    else:
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            model = Sequential()
            model.add(Input(shape=(28, 28, 1)))
            model.add(Lambda(standardize))
            model.add(Convolution2D(hyperparameters['conv_filters'][0], (3, 3), activation='relu'))
            model.add(Convolution2D(hyperparameters['conv_filters'][0], (3, 3), activation='relu'))
            model.add(MaxPooling2D())
            model.add(Dropout(hyperparameters['dropout_rate']))
            model.add(Convolution2D(hyperparameters['conv_filters'][1], (3, 3), activation='relu'))
            model.add(Convolution2D(hyperparameters['conv_filters'][1], (3, 3), activation='relu'))
            model.add(MaxPooling2D())
            model.add(Dropout(hyperparameters['dropout_rate']))
            model.add(Flatten())
            model.add(Dense(hyperparameters['dense_units'], activation='relu'))
            model.add(Dropout(hyperparameters['dropout_rate']))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer=Adam(
                learning_rate=hyperparameters['learning_rate']), metrics=['accuracy'])

    batches, test_batches = generate_batches(
        x_train, y_train, hyperparameters['batch_size'], hyperparameters['validation_split'])

    history = model.fit(batches, epochs=4, validation_data=test_batches,
                        steps_per_epoch=batches.n, validation_steps=test_batches.n, verbose=0)

    accuracy = max(history.history['val_accuracy'])

    return model, accuracy, hyperparameters


def generate_hyperparameter_combinations():
    """Generate all possible combinations of hyperparameters."""
    keys = CNN_HYPERPARAMETERS.keys()
    values = CNN_HYPERPARAMETERS.values()

    combinations = []
    for combination in itertools.product(*values):
        hyperparameter_dict = dict(zip(keys, combination))
        combinations.append(hyperparameter_dict)

    return combinations


# Load data and configure for GPU
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    x_train, y_train, x_test = load_data()

    # Feature standardization
    mean = x_train.mean().astype(np.float32)
    std = x_train.std().astype(np.float32)


def standardize(x):
    return (x - mean) / std


hyperparameter_combinations = generate_hyperparameter_combinations()
num_combinations = len(hyperparameter_combinations) // size
start = rank * num_combinations
end = len(hyperparameter_combinations) if rank == size - 1 else start + num_combinations
my_combinations = hyperparameter_combinations[start:end]

best_accuracy = -1
best_model = None
best_hyperparameters = None

if rank == 0:
    print(f"Training {len(my_combinations)} hyperparameter combinations per process on GPU...")

for i, hyperparameters in enumerate(tqdm(my_combinations, desc=f'Process {rank} (GPU)', position=rank, leave=True)):
    model, accuracy, params = cnn_model(x_train, y_train, hyperparameters)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_hyperparameters = params

all_accuracies = comm.gather(best_accuracy, root=0)  # Blocking
all_hyperparameters = comm.gather(best_hyperparameters, root=0)  # Blocking
# Do not gather all models to avoid memory issues

if rank == 0:
    global_best_accuracy = max(all_accuracies)
    best_rank = all_accuracies.index(global_best_accuracy)
    best_hyperparameters = all_hyperparameters[best_rank]

    print(f"Global best accuracy: {global_best_accuracy:.4f} found by process {best_rank}")
    print(f"Best hyperparameters: {best_hyperparameters}")

    # Signal the best process to save its model
    for i in range(size):
        if i == best_rank:
            comm.send(True, dest=i, tag=33)
        else:
            comm.send(False, dest=i, tag=33)

else:
    should_save = comm.recv(source=0, tag=33)

file = f'{out_dir}/best_model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'

if rank == 0 and best_rank == 0:
    best_model.save(file)
elif rank != 0 and should_save:
    best_model.save(file)

if rank == 0:
    print("GPU-accelerated training completed!")
