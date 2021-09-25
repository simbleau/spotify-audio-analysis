#!/usr/bin/python

import itertools
from spotify_audio_analysis import run
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
def grid_search(endpoint, layer_types, layer_counts, neuron_counts, loss_functions):
    if PRINT_PERMUTATIONS:
        amt_loss_functions = len(loss_functions)
        amt_layer_counts = len(layer_counts)
        amt_layer_types = len(layer_types)
        amt_neuron_counts = len(neuron_counts)
        amt_total = 0
        for layer_count in layer_counts:
            amt_total += layer_count ** amt_neuron_counts * amt_layer_types
        amt_total *= amt_loss_functions
        print(f"Total permutations: {amt_total}")

    layer_permutations = []
    print("Calcuating permutations...")
    for loss_function in loss_functions:
        for layer_count in layer_counts:
            neuron_count_permutations = list(itertools.product(neuron_counts, repeat=layer_count))
            neuron_activation_permutations = list(itertools.product(neuron_count_permutations, layer_types))
            amt_total_perms = len(neuron_activation_permutations)
            print(f"For layer count {layer_count},  total permutations discovered: {amt_total_perms}")
            for layer_neuron_counts, activation in neuron_activation_permutations:
                layers = []
                for neuron_amt in layer_neuron_counts:
                    layers.append(Dense(neuron_amt, activation=activation))
            layer_permutations.append(layers)
    print("All permutations compiled.")

    if RANDOM_ORDERING:
        print("Randomizing permutation order...")
        shuffle(layer_permutations)
        print("Randomized.")

    print("Beginning grid search...")
    for layer_permutation in layer_permutations:
        run(endpoint, layer_permutation, loss_function, optimizer, batch_size, epochs, True)
    print("Grid search complete.")
