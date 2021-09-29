#!/usr/bin/python
from copy import copy, deepcopy

from helper_methods import *

import shutil
import tensorflow
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from os.path import exists
import numpy as np
from sklearn.model_selection import train_test_split


def run(endpoint, layers, loss_function, optimizer, batch_size, epochs, save):
    # Clear backend
    keras.backend.clear_session()

    # Printing Debug information
    num_gpus = len(tensorflow.config.experimental.list_physical_devices('GPU'))
    using_gpus = num_gpus >= 1
    print(f"Using GPU: {using_gpus}\n")

    # Debug variables
    layers_str = "[" + "|".join(str(str(x.units) + " " + x.activation._keras_api_names[0][18:]) for x in layers) + "]"
    loss_function_name = loss_function.name
    print(f"{endpoint} hyper-parameters:\n\t" +
          f"Layers: {layers_str}\n\tLoss Function: {loss_function_name}\n\tBatch Size: {batch_size}\n\t" +
          f"Epochs: {epochs}")

    # Begin run

    # Setup path for artifacts
    output_path = f'checkpoints/{endpoint}-artifacts'
    # Prune previous attempts
    if exists(output_path):
        shutil.rmtree(output_path)

    # Get x, y
    x_spotify_train, y_spotify_train = get_xy('data/spotify_train.npz', endpoint)
    x_spotify_valid, y_spotify_valid = get_xy('data/spotify_valid.npz', endpoint)

    x = np.concatenate([x_spotify_train,x_spotify_valid])
    y = np.concatenate([y_spotify_train,y_spotify_valid])

    x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size=0.2)

    # Setup callbacks
    model_checkpoint = setup_model_checkpoints(output_path, save_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # Sequential Model
    model = Sequential()
    model.add(Input(129, name='input'))
    # Hidden Layers
    for layer in layers:
        model.add(layer)
    # Add output layer
    if endpoint == "timbre":
        # Output layer should allows negative values
        model.add(Dropout(0.2))
        model.add(Dense(12, activation='linear', name='output'))
    elif endpoint == "pitch":
        # Output layer clamps values between 0 and 1
        model.add(Dropout(0.2))
        model.add(Dense(12, activation='softmax', name='output'))
    elif endpoint == "loudness":
        # Output layer should allows negative values
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear', name='output'))
    else:
        print("This shouldn't happen!")
        exit(1)

    model.optimizer = optimizer
    model.compile(loss=loss_function)
    model.summary()

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=[x_valid, y_valid],
                        callbacks=[model_checkpoint, early_stopping])
    number_of_epochs_ran = len(history.history['val_loss'])
    val_loss = model.evaluate(x_valid, y_valid, verbose=0)

    # Write result to results
    csv_result = f"{endpoint},{layers_str},{loss_function_name},{batch_size},{epochs},{number_of_epochs_ran},{val_loss}\n"
    file1 = open('results.csv', 'a+')
    file1.write(csv_result)
    file1.close()
    print("Results appended.\n")

    # At the end, get the best model and visualize it.
    model, best_epoch, best_loss = get_best_model(output_path)
    print(f"The best model was discovered on epoch {best_epoch} and had a loss of {best_loss}")

    # Check if the current model is better or not
    prev_best_val_loss = float('inf')
    if exists(f"best_models/{endpoint}.h5"):
        prev_best_model = keras.models.load_model(f"best_models/{endpoint}.h5", compile=True,
                                                  custom_objects={'Normalization': Normalization})
        prev_best_val_loss = prev_best_model.evaluate(x_valid, y_valid, verbose=0)
    if prev_best_val_loss - best_loss > 0.000001:
        print(f"NEW RECORD! Loss: {best_loss}, saved to: best_models/{endpoint}.h5")
        model.save(f"best_models/{endpoint}.h5")
    else:
        print(f"This run did not beat the previous best loss of {prev_best_val_loss}")

    print("Saving visualizations of the best model...")

    # Save result
    if save:
        # Save visualizations of the best model
        visualize(model, x=x_train, y_true=y_train, endpoint=endpoint, name='Training', output_path=output_path)
        print(f"Saved " + str(os.path.join(output_path, f'visualize_Training.png')) + '.')

        training_png_file_path = os.path.join(output_path, f'visualize_Validation.png')
        visualize(model, x=x_valid, y_true=y_valid, endpoint=endpoint, name='Validation', output_path=output_path)
        print(f"Saved " + str(os.path.join(output_path, f'visualize_Validation.png')) + '.')
