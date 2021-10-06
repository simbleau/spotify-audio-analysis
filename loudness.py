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
    Dense(250, activation='relu'),
    Dense(250, activation='linear'),
    Dense(600, activation='relu'),
    Dense(600, activation='linear'),
]

# Loss function
loss_function = MeanSquaredError()

# Optimizers
optimizer = Adamax(learning_rate=1)

# Batch Size
batch_size = 512

# Epochs
epochs = 100

# Run
if __name__ == '__main__':
    run("loudness", layers, loss_function, optimizer, batch_size, epochs, True)
