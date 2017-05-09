from __future__ import absolute_import

import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from load_data import extract_data, generator, parse_log
from model import build_model, build_model_inception, build_model_lenet, build_model_nvidia

DEFAULT_MODEL_PATH = 'model.h5'


def fit_model(model, X_train, y_train, epochs, save_path=DEFAULT_MODEL_PATH):
    """Fit model to data."""
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        shuffle=True,
        nb_epoch=epochs
    )
    if save_path:
        model.save(save_path)


def fit_generator(model,
                  train_lines, train_generator,
                  validation_lines, validation_generator,
                  epochs=5,
                  save_path=DEFAULT_MODEL_PATH):
    """Fit model to data using generators"""
    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        samples_per_epoch=(len(train_lines) * 2),  # for flipped images
        nb_val_samples=(len(validation_lines) * 2),  # for flipped images
        nb_epoch=epochs
    )
    if save_path:
        model.save(save_path)


def fit_async(model, data_dirs=[], epochs=5, header=False,
              save_path=DEFAULT_MODEL_PATH):
    """Fit using generators"""
    lines = []

    for _dir in data_dirs:
        log_path = './{}/driving_log.csv'.format(_dir)
        _lines = parse_log(log_path)

        if header:
            _lines = _lines[1:]

        lines += _lines

    # Split into train, validation
    lines = shuffle(lines)
    train_lines, validation_lines = train_test_split(lines, test_size=0.2)

    # Create generators
    train_gen = generator(train_lines)
    validation_gen = generator(validation_lines)

    fit_generator(
        model,
        train_lines=train_lines,
        train_generator=train_gen,
        validation_lines=validation_lines,
        validation_generator=validation_gen,
        epochs=epochs,
        save_path=save_path
    )


def fit_sync(model, data_dirs=[], epochs=5, save_path=DEFAULT_MODEL_PATH):
    """Fit without generators"""
    X_train, y_train = [], []

    for _dir in data_dirs:
        log_path = './{}/driving_log.csv'.format(_dir)
        images, angles = extract_data(log_path)

        X_train.append(images)
        y_train.append(angles)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    fit_model(
        model,
        X_train, y_train,
        epochs=epochs,
        save_path=save_path
    )


# Load data and train model
if __name__ == '__main__':

    # Params
    epochs = 3

    # Get logs
    data_dirs = [
        'data-center',
        'data-reverse',
        'data-recovery',
    ]

    # Run different combination of models
    for _model in ['lenet', 'nvidia']:

        for dropout in [0, 0.25, 0.5]:

            for reg_beta in [0, 5e-4]:

                if _model == 'lenet':
                    model = build_model_lenet(
                        dropout=dropout,
                        reg_beta=reg_beta
                    )
                elif _model == 'nvidia':
                    model = build_model_nvidia(
                        dropout=dropout,
                        reg_beta=reg_beta
                    )

                save_path = '{}_{}_{}.h5'.format(
                    _model, dropout, reg_beta
                )
                print('Training {} model with params'.format(_model))
                print('- dropout: {}'.format(dropout))
                print('- reg_beta: {}'.format(reg_beta))
                print('- save_path: {}'.format(save_path))
                model.summary

                fit_async(
                    model, data_dirs,
                    epochs=epochs, save_path=save_path
                )
