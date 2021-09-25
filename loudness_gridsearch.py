#!/usr/bin/python

from gridsearch import grid_search
from tensorflow.keras.losses import *

# Hyper-parameters for Grid Search
layer_types = ['tanh', 'sigmoid', 'linear']
layer_counts = [1, 2, 3, 4]
neuron_counts = [10, 50, 100, 500]
loss_functions = [MeanSquaredError()]

# Run grid search
if __name__ == "__main__":
    grid_search("loudness", layer_types, layer_counts, neuron_counts, loss_functions)
