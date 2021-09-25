#!/usr/bin/python

import sys
from spotify_audio_analysis import run
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *


# Model
#
# Example Output (from https://cs.appstate.edu/~rmp/cs5440/AnalyzeDocumentation.pdf)
# "loudness": -60.000
layers = [
    Dense(500, activation='tanh'),
]

# Loss function
loss_function = MeanSquaredError()

# Optimizers
optimizer = Adam()

# Batch Size
batch_size = 1024

# Epochs
epochs = 20

# Run
if __name__ == '__main__':
    run("loudness", layers, loss_function, optimizer, batch_size, epochs, True)
