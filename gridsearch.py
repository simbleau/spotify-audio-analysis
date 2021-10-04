#!/usr/bin/python

import itertools
from spotify_audio_analysis import run,run_with_cross_validation
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from random import shuffle

# Constants - (ACCEPTABLE ERROR)
optimizer = Adamax(learning_rate=0.01)
batch_size = 1024
epochs = 100

# Debug settings
PRINT_PERMUTATIONS = True  # Whether to print the amount of permutations while running
RANDOM_ORDERING = True  # Whether to grid search in random order (good for faster discovery)

# Run grid search
def grid_search(endpoint, layer_types, layer_counts, neuron_counts, loss_functions, folds = 5,run_with_KFolds = False):
    if PRINT_PERMUTATIONS:
        amt_loss_functions = len(loss_functions)
        amt_layer_types = len(layer_types)
        amt_neuron_counts = len(neuron_counts)
        amt_total = 0
        for layer_count in layer_counts:
            amt_neuron_total = amt_neuron_counts ** layer_count
            amt_activation_total = amt_layer_types ** layer_count
            amt_total += (amt_neuron_total * amt_activation_total)
        amt_total *= amt_loss_functions
        print(f"Total permutations: {amt_total}")

    layer_permutations = []
    print("Calcuating permutations...")
    for loss_function in loss_functions:
        for layer_count in layer_counts:     
            neuron_count_permutations = list(itertools.product(neuron_counts, repeat=layer_count))
            neuron_activation_permutations = list(itertools.product(layer_types, repeat=layer_count))      
            perms = list(itertools.product(neuron_count_permutations, neuron_activation_permutations))
            amt_total_perms = len(perms)
            for layer_neuron_counts, activations in perms:
                layers = []
                for i in range(len(layer_neuron_counts)):
                    neuron_amt = layer_neuron_counts[i]
                    activation = activations[i]
                    layer_name = "layer" + str(len(layers))
                    layers.append(Dense(neuron_amt, activation=activation, name=layer_name))
                layer_permutations.append(layers)
    amt_layer_permutations = len(layer_permutations)
    print(f"All {amt_layer_permutations} permutations compiled.")


    if RANDOM_ORDERING:
        print("Randomizing permutation order...")
        shuffle(layer_permutations)
        print("Randomized.")
    print("Beginning grid search...")
    for layer_permutation in layer_permutations:
        if run_with_KFolds:
            run_with_cross_validation(endpoint, layer_permutation, loss_function, optimizer, batch_size, epochs, False, folds=folds)
        else:
            run(endpoint, layer_permutation, loss_function, optimizer, batch_size, epochs, False)
    print("Grid search complete.")
