#!/usr/bin/python

import sys
from spotify_audio_analysis import run, run_with_cross_validation
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *


# Model
#
# Example Output (from https://cs.appstate.edu/~rmp/cs5440/AnalyzeDocumentation.pdf)
# "loudness": -60.000
layers = [
    Dense(750, activation='relu'),
    Dense(750, activation='relu'),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
]

# Loss function
loss_function = MeanSquaredError()

# Optimizers
optimizer = Adamax(learning_rate=1)

# Batch Size
batch_size = 1024

# Epochs
epochs = 10000

# Run
if __name__ == '__main__':
    run_with_cross_validation("loudness", layers, loss_function, optimizer, batch_size, epochs,
                              folds=5, patience=100)
