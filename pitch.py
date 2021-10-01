#!/usr/bin/python

import sys
from spotify_audio_analysis import run_with_cross_validation
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *


# Model
#
# Example Output (from https://cs.appstate.edu/~rmp/cs5440/AnalyzeDocumentation.pdf)
# "pitches": [0.370, 0.067, 0.055, 0.073, 0.108, 0.082, 0.123, 0.180, 0.327, 1.000, 0.178, 0.234]
#[2000 sigmoid|1250 sigmoid|2500 sigmoid|1000 sigmoid]
layers = [
    Dense(750, activation='sigmoid'),
    Dense(2500, activation='sigmoid'),
    Dense(500, activation='sigmoid')
]

# Loss function
loss_function = MeanSquaredLogarithmicError()

# Optimizers
optimizer = Adamax(learning_rate=1)

# Batch Size
batch_size = 4098

# Epochs
epochs = 1

# Run
if __name__ == '__main__':
    run_with_cross_validation("pitch", layers, loss_function, optimizer, batch_size, epochs, folds=2)
