#!/usr/bin/python

import sys
from spotify_audio_analysis import run
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *

OUTPUT_SIZE = 12

# Model
layers = [
    Dense(OUTPUT_SIZE, activation='linear')
]

# Loss function
loss_function = MeanSquaredError()

# Optimizers
optimizer = Adamax(learning_rate=0.01)

# Batch Size
batch_size = 512

# Epochs
epochs = 1

# Run
if __name__ == '__main__':
    run("pitch", layers, loss_function, optimizer, batch_size, epochs, True)