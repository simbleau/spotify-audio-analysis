import matplotlib.pyplot as plt
import io

import numpy as np
import tensorflow as tf
import sys


timbre_model_path = 'best_models/timbre.h5'
pitch_model_path = 'best_models/pitch.h5'
loudness_model_path = 'best_models/loudness.h5'


# Function that will print the summary of a h5 file and model parameters,
# Usage is to find parameters of a previous model.
def model_sum(filepath):
    model = tf.keras.models.load_model(filepath)
    s = io.StringIO()
    model.summary(print_fn=lambda x: s.write(x + '\n'))
    model_summary = s.getvalue()
    # history = model.fit(np.arrange(100).reshape(5, 20), np.zeros(5), epochs=10)
    s.close()
    print("Filepath: " + filepath)
    print("The model summary is:\n\n{}".format(model_summary))
    # TODO: Add plotting and model analysis
    # print("Model Evaluation: ")
    # print(model.evaluate())
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # plt.plot(history.history['loss'], label='(training data)')
    # plt.plot(history.history['val_loss'], label='(validation data)')
    # plt.title('Spotify Analysis for ' + filepath)
    # plt.ylabel('MAE value')
    # plt.xlabel('No. epoch')
    # plt.legend(loc="upper left")
    # plt.show()


# Running this file will require you to type a filepath arg
# or Summarize a specific model category in best_models
# Usage: model_sum filepath
if __name__ == '__main__':
    value = input("Enter / type a filepath arg "
                  "or a specific model category"
                  "Usage: pitch / loudness / timbre / filepath\n")
    if len(value) < 1:
        print("Usage: model_sum filepath "
              "\n OR \n model_sum pitch / timbre / loudness \n"
              " ""For the best model of each category in best_models\n")
        sys.exit(1)
    elif value == "pitch":
        model_sum(pitch_model_path)
    elif value == "timbre":
        model_sum(timbre_model_path)
    elif value == "loudness":
        model_sum(loudness_model_path)
