#!/usr/bin/python

import sys
from copy import deepcopy

from spotify_audio_analysis import run, run_with_cross_validation
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *


def tune(endpoint):
    """
    Tunes a model's hyper-parameters by small delta improvements until convergence.
    """
    print(f"Tuning {endpoint} model.")
    if endpoint == "pitch":
        from pitch import layers
        from pitch import loss_function
        from pitch import optimizer
        from pitch import batch_size
    elif endpoint == "timbre":
        from timbre import layers
        from timbre import loss_function
        from timbre import optimizer
        from timbre import batch_size
    elif endpoint == "loudness":
        from loudness import layers
        from loudness import loss_function
        from loudness import optimizer
        from loudness import batch_size
    else:
        print("Unknown model. Exiting...")
        sys.exit(1)

    # Additional hyper-parameters to try
    all_loss_functions = [MeanSquaredError(), MeanAbsoluteError(), MeanSquaredLogarithmicError(), Huber()]
    all_optimizers = [Adam(), Adamax(learning_rate=0.1), SGD()]

    # The best loss to be tracked
    best_loss = float('inf')

    # Discover best loss function
    print("Discovering best loss function...")
    for lf in all_loss_functions:
        new_loss = run(endpoint, layers, lf, optimizer, batch_size, 10000, False)
        if new_loss < best_loss:
            # Overwrite loss function
            loss_function = lf
            print(f"New best loss-function discovered: {lf}")

    # Discover best optimizer
    print("Discovering best optimizer...")
    for op in all_optimizers:
        new_loss = run(endpoint, layers, loss_function, op, batch_size, 10000, False)
        if new_loss < best_loss:
            # Overwrite optimizer
            optimizer = op
            print(f"New best optimizer discovered: {lf}")

    # Discover best batch sizes
    print("Discovering best batch size...")
    max_patience = 10
    patience = 0
    step_size = 100

    best_batch_size = batch_size
    best_loss = run(endpoint, layers, loss_function, optimizer, batch_size, 10000, False)
    # Attempt to go up
    high_batch_size = float('inf'), batch_size
    while patience < max_patience:
        # Increase batch size
        high_batch_size += step_size
        new_loss = run(endpoint, layers, loss_function, optimizer, high_batch_size, 10000, False)
        if new_loss < best_loss:
            patience = 0
            best_batch_size = high_batch_size
            best_loss = new_loss
            print(f"New best batch-size discovered: {high_batch_size}")
        else:
            patience += 1
    # Attempt to go down
    low_batch_size = float('inf'), batch_size
    while patience < max_patience:
        # Decrease batch size
        low_batch_size -= max(1, step_size)
        new_loss = run(endpoint, layers, loss_function, optimizer, low_batch_size, 10000, False)
        if new_loss < best_loss:
            patience = 0
            best_batch_size = low_batch_size
            best_loss = new_loss
            print(f"New best batch-size discovered: {lf}")
        else:
            patience += 1
    # Overwrite batch size
    batch_size = best_batch_size

    # Discover best neuron counts with passes
    print("Discovering best neuron counts...")
    passes = 1
    max_patience = 10
    patience = 0
    step_size = 100

    best_loss = run(endpoint, layers, loss_function, optimizer, batch_size, 10000, False)
    for pass_num in range(passes):
        for i in range(len(layers)):
            layers_copy = deepcopy(layers)
            layer = layers_copy[i]

            original_count = layer.units
            # Attempt to go up
            while patience < max_patience:
                # Increase neuron count
                layer.units += step_size
                new_loss = run(endpoint, layers_copy, loss_function, optimizer, high_batch_size, 10000, False)
                if new_loss < best_loss:
                    patience = 0
                    best_loss = new_loss
                    layers[i].units = layer.units # Modify original set
                    print(f"New best layer-count discovered: {layer.units}")
                else:
                    patience += 1
            # Reset
            layer.units = original_count
            # Attempt to go down
            while patience < max_patience:
                # Decrease neuron count
                layer.units -= step_size
                new_loss = run(endpoint, layers_copy, loss_function, optimizer, high_batch_size, 10000, False)
                if new_loss < best_loss:
                    patience = 0
                    best_loss = new_loss
                    layers[i].units = layer.units # Modify original set
                    print(f"New best layer-count discovered: {layer.units}")
                else:
                    patience += 1
    
    # TODO: Dropout

    # Conducts a final run with no patience to receive as much progress as possible.
    run_with_cross_validation(endpoint, layers, loss_function, optimizer, batch_size, 10000, False,
                              folds=5,
                              patience=15)


# Run
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: tune <timbre|pitch|loudness>")
        sys.exit(1)
    else:
        tune(sys.argv[1])
