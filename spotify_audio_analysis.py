#!/usr/bin/python

from helper_methods import *

import tensorflow
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from os.path import exists


def run(endpoint, layers, loss_function, optimizer, batch_size, epochs, save):
    # Printing Debug information
    num_gpus = len(tensorflow.config.experimental.list_physical_devices('GPU'))
    using_gpus = num_gpus >= 1
    print(f"Using GPU: {using_gpus}\n")

    # Debug variables
    layers_str = "[" + ",".join(str(x.units) for x in layers) + "]"
    loss_function_name = loss_function.name
    print(f"{endpoint} hyper-parameters:\n\t" +
          f"Layers: {layers_str}\n\tLoss Function: {loss_function_name}\n\tBatch Size: {batch_size}\n\t" +
          f"Epochs: {epochs}")

    # Begin run

    # Clear backend
    keras.backend.clear_session()

    # Setup path for artifacts
    output_path = f'checkpoints/{endpoint}-artifacts'

    # Get x, y
    x_train, y_train = get_xy('data/spotify_train.npz', endpoint)
    x_valid, y_valid = get_xy('data/spotify_valid.npz', endpoint)

    # Setup callbacks
    model_checkpoint = setup_model_checkpoints(output_path, save_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # Sequential Model
    model = Sequential()
    model.add(Input(129))
    # Hidden Layers
    for layer in layers:
        model.add(layer)

    model.optimizer = optimizer
    model.compile(loss='mean_squared_error')
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
    visualize(model, x=x_train, y_true=y_train, endpoint=endpoint, name='Training', output_path=output_path)
    visualize(model, x=x_valid, y_true=y_valid, endpoint=endpoint, name='Validation', output_path=output_path)
    print(f"The best model was discovered on epoch {best_epoch} and had a loss of {best_loss}")

    # Save result
    if save:
        # Check if the current model is better or not
        prev_best_val_loss = float('inf')
        if exists(f"best_models/{endpoint}.h5"):
            prev_best_model = keras.models.load_model(f"best_models/{endpoint}.h5", compile=True,
                                                      custom_objects={'Normalization': Normalization})
            prev_best_val_loss = prev_best_model.evaluate(x_valid, y_valid, verbose=0)
        if best_loss < prev_best_val_loss:
            print(f"NEW RECORD! BEST MODEL SAVED: best_models/{endpoint}.h5")
            model.save(f"best_models/{endpoint}.h5")
        else:
            print(f"This run did not surpass previous results.")
