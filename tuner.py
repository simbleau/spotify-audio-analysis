#!/usr/bin/python

import sys
from spotify_audio_analysis import run
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
    all_loss_functions = [MeanSquaredError(), MeanAbsoluteError(), MeanSquaredLogarithmicError()]
    all_optimizers = [Adam(), Adamax(learingrate=0.1), SGD()]

    # The best loss to be tracked
    best_loss = float('inf')

    # Discover best loss function
    for lf in all_loss_functions:
        new_loss = run(endpoint, layers, lf, optimizer, batch_size, 10000, False)
        if new_loss < best_loss:
            # Overwrite loss function
            loss_function = lf
            print(f"New best loss-function discovered: {lf}")

    # Discover best optimizer
    for op in all_optimizers:
        new_loss = run(endpoint, layers, loss_function, op, batch_size, 10000, False)
        if new_loss < best_loss:
            # Overwrite optimizer
            optimizer = op
            print(f"New best optimizer discovered: {lf}")

    # Discover best batch sizes
    max_patience = 10
    patience = 0
    step_size = 100

    best_batch_size = batch_size
    best_loss = run(endpoint, layers, loss_function, optimizer, batch_size, 10000, False)
    # Attempt to go up
    high_best_loss, high_batch_size = float('inf'), batch_size
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
    low_best_loss, low_batch_size = float('inf'), batch_size
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


    # TODO: Discover best neuron counts with passes


    # Conducts a final run with no patience to receive as much progress as possible.
    run(endpoint, layers, loss_function, optimizer, batch_size, 10000, False, patience=100)


# Run
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: tune <timbre|pitch|loudness>")
        sys.exit(1)
    else:
        tune(sys.argv[1])
