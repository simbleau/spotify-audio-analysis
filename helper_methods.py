import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow_core.python.keras.layers.preprocessing.normalization import Normalization
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.losses import mean_squared_error, kullback_leibler_divergence
import tensorflow as tf


def get_xy(npz_file, endpoint='pitch'):
    """
    Gets the data matrix, X, and target, y for the desired 'endpoint'.

    :param npz_file: The path to the npz file
    :param endpoint: 'pitch', 'timbre', or 'loudness'
    :return:
    """
    with np.load(npz_file) as data:
        x = data['x']
        key = f'y_{endpoint}'
        if key not in data:
            raise ValueError(f'Unknown endpoint {endpoint}')
        y = data[key]
        if y.ndim == 1:
            y = data[key].reshape((-1, 1))
    return x, y


def setup_model_checkpoints(output_path, save_freq):
    """
    Setup model checkpoints using the save path and frequency.

    :param output_path: The directory to store the checkpoints in
    :param save_freq: The frequency with which to save them "epoch" means each epoch
                      See ModelCheckpoint documentation.
    :return: a ModelCheckpoint
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_checkpoint = ModelCheckpoint(
        os.path.join(output_path, 'model.{epoch:05d}_{val_loss:f}.h5'),
        save_weights_only=False,
        save_freq=save_freq,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    return model_checkpoint


def visualize(model, x, y_true, endpoint='loudness', name='', output_path=''):
    """
    Create a joint distribution plot that shows relationship between
    model estimates and true values.

    :param model: A trained model
    :param x: The data matrix, X
    :param y_true: The target vector or matrix, y
    :param name: The name for the figure
    :param output_path: The output directory to save the PNG
    :return: None
    """
    png_file = os.path.join(output_path, f'visualize_{name}.png')
    y_pred = model.predict(x)

    # check assumptions
    if y_true.shape != y_pred.shape:
        print(f'WARNING: output should have shape {y_true.shape} not {y_pred.shape}. Broadcasting output.')
        y_pred = np.broadcast_to(y_pred, y_true.shape)

    # loss should be mean squared error unless pitch target
    metric = mean_squared_error

    if endpoint == 'pitch':
        # if each row of target sums to one and is nonnegative, we know it must be pitches.
        metric = kullback_leibler_divergence
        if not (y_pred >= 0).all():
            print(f'WARNING: output should be nonnegative. Setting negative values to a small positive value.')
            y_pred[y_pred < 0] = 1e-6
        if not np.allclose(y_pred.sum(axis=1), 1):
            print(f'WARNING: outputs should sum to one. Scaling rows of output to sum to one.')
            y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

    loss = tf.reduce_mean(metric(y_true, y_pred)).numpy()

    # make joint plot
    jg = sns.jointplot(x=y_true.reshape((-1,)), y=y_pred.reshape((-1,)), kind='hist')
    jg.fig.suptitle(f'{name} (loss = {loss:.6f})')
    jg.set_axis_labels(xlabel='Actual', ylabel='Model')
    max_value = max(y_pred.max(), y_true.max())
    min_value = min(y_pred.min(), y_true.min())
    jg.ax_joint.plot([min_value, max_value], [min_value, max_value], color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig(png_file)
    plt.close(jg.fig)


def get_best_model(output_path):
    """
    Parses the output_path to find the best model. Relies on the ModelCheckpoint
    saving a file name with the validation loss in it. If a model was saved with
    a Normalization layer, it's provided as a custom object.

    :param output_path: The directory to scan for H5 files
    :return: The best model compiled.
    """
    min_loss = float('inf')
    best_model_file = None
    best_epoch = None
    for file_name in os.listdir(output_path):
        if file_name.endswith('.h5'):
            val_loss = float('.'.join(file_name.split('_')[1].split('.')[:-1]))
            epoch = int(file_name.split('.')[1].split('_')[0])
            if val_loss < min_loss:
                best_model_file = file_name
                min_loss = val_loss
                best_epoch = epoch
    print(f'loading best model: {best_model_file}')
    model = keras.models.load_model(os.path.join(output_path, best_model_file), compile=True,
                                    custom_objects={'Normalization': Normalization})
    return model, best_epoch, min_loss