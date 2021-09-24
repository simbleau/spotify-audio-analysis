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
# "pitches": [0.370, 0.067, 0.055, 0.073, 0.108, 0.082, 0.123, 0.180, 0.327, 1.000, 0.178, 0.234]
layers = [
    Dense(500, activation='tanh'),
    Dense(500, activation='sigmoid'),
    Dense(OUTPUT_SIZE, activation='softmax') # Should be an activation which clamps values between 0 and 1
]

# Loss function
loss_function = MeanSquaredError()

# Optimizers
optimizer = Adamax(learning_rate=0.1)

# Batch Size
batch_size = 4098

# Epochs
epochs = 30

# Run
if __name__ == '__main__':
    run("pitch", layers, loss_function, optimizer, batch_size, epochs, True)