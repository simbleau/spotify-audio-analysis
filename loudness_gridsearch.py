#!/usr/bin/python

from gridsearch import grid_search
from tensorflow.keras.losses import *

# Hyper-parameters for Grid Search
layer_types = ['relu', 'linear']
layer_counts = [3, 4]
neuron_counts = [250, 400, 600, 750]
loss_functions = [MeanSquaredError(), MeanAbsoluteError(), MeanSquaredLogarithmicError()]
run_with_Kfolds = True
folds=5

# Run grid search
if __name__ == "__main__":
    grid_search("loudness", layer_types, layer_counts, neuron_counts, loss_functions, folds,run_with_Kfolds)
