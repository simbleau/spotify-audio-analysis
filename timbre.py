#!/usr/bin/python

import sys
from spotify_audio_analysis import run, run_with_cross_validation
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *

# Model
#
# Example Output (from https://cs.appstate.edu/~rmp/cs5440/AnalyzeDocumentation.pdf)
# "timbre": [24.736, 110.034, 57.822, -171.580, 92.572, 230.158, 48.856, 10.804, 1.371, 41.446, -66.896, 11.207]
layers = [
    Dense(1500, activation='sigmoid'),
    Dense(1200, activation='sigmoid'),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu')
]

# Loss function
loss_function = MeanSquaredError()

# Optimizers
optimizer = Adamax(learning_rate=0.1)

# Batch Size
batch_size = 1024

# Epochs
epochs = 10000

# Run
if __name__ == '__main__':
    run_with_cross_validation("timbre", layers, loss_function, optimizer, batch_size, epochs,
                              folds=5, patience=100)
