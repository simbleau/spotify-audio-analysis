#!/usr/bin/python

import sys
from spotify_audio_analysis import run
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *

OUTPUT_SIZE = 1

# Model
layers = [
    Dense(500, activation='tanh'),
    Dense(OUTPUT_SIZE, activation='linear')
]

# Loss function
loss_function = MeanSquaredError()

# Optimizers
optimizer = Adam()

# Batch Size
batch_size = 64

# Epochs
epochs = 1

# Run
if __name__ == '__main__':
    run("loudness", layers, loss_function, optimizer, batch_size, epochs, True)
