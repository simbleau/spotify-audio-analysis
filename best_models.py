#!/usr/bin/python

from helper_methods import *

from os.path import exists


def print_best_model(path, endpoint):
    # Get best model and print out the loss
    if exists(path):
        # Get validation data
        x_valid, y_valid = get_xy('data/spotify_valid.npz', endpoint)
        model = keras.models.load_model(path, compile=True, custom_objects={'Normalization': Normalization})
        val_loss = model.evaluate(x_valid, y_valid, verbose=0)
        print(f"{endpoint}: {val_loss} loss")
    else:
        print(f"No best model for {endpoint}")


def main():
    # Setup path for artifacts
    timbre_model_path = f'best_models/timbre.h5'
    pitch_model_path = f'best_models/pitch.h5'
    loudness_model_path = f'best_models/loudness.h5'

    print_best_model(timbre_model_path, 'timbre')
    print_best_model(pitch_model_path, 'pitch')
    print_best_model(loudness_model_path, 'loudness')


if __name__ == "__main__":
    main()
