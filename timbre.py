#!/usr/bin/python

import sys
from spotify_audio_analysis import run
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *

OUTPUT_SIZE = 12

# Model
#
# Example Output (from https://cs.appstate.edu/~rmp/cs5440/AnalyzeDocumentation.pdf)
# "timbre": [24.736, 110.034, 57.822, -171.580, 92.572, 230.158, 48.856, 10.804, 1.371, 41.446, -66.896, 11.207]
layers = [
    Dense(500, activation='linear'),
    Dense(OUTPUT_SIZE, activation='linear') # Should be an activation which allows negative values
]

# Loss function
loss_function = MeanSquaredError()

# Optimizers
optimizer = Adamax(learning_rate=0.1)

# Batch Size
batch_size = 2048

# Epochs
epochs = 100

# Run
if __name__ == '__main__':
    run("timbre", layers, loss_function, optimizer, batch_size, epochs, True)