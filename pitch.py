#!/usr/bin/python

import sys
from spotify_audio_analysis import run
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *

OUTPUT_SIZE = 12

# Model
layers = [
    Dense(500, activation='sigmoid'),
    Dense(OUTPUT_SIZE, activation='softmax')
]

# Loss function
loss_function = MeanSquaredError()

# Optimizers
optimizer = Adamax(learning_rate=0.1)

# Batch Size
batch_size = 2048

# Epochs
epochs = 20

# Run
if __name__ == '__main__':
    run("pitch", layers, loss_function, optimizer, batch_size, epochs, True)