#!/usr/bin/python

from gridsearch import grid_search
from tensorflow.keras.losses import *

# Hyper-parameters for Grid Search
layer_types = ['sigmoid']
layer_counts = [3,4, 5]
neuron_counts = [100, 500, 750, 1000]
loss_functions = [MeanSquaredError()]

# Run grid search
if __name__ == "__main__":
    grid_search("pitch", layer_types, layer_counts, neuron_counts, loss_functions)
