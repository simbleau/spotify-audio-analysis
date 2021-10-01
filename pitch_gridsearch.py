#!/usr/bin/python

from gridsearch import grid_search
from tensorflow.keras.losses import *

# Hyper-parameters for Grid Search
layer_types = ['tanh', 'sigmoid', 'sigmoid', 'relu', 'linear']
layer_counts = [1, 2]
neuron_counts = [10, 50, 100, 500]
loss_functions = [MeanSquaredError()]
run_with_Kfolds = False
folds = 5

# Run grid search
if __name__ == "__main__":
    grid_search("pitch", layer_types, layer_counts, neuron_counts, loss_functions, folds, run_with_Kfolds)
