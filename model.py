import csv
import sys

import numpy as np

import cv2
from keras.layers import Dense, Flatten, Lambda
from keras.models import Sequential

DEFAULT_LOG_PATH = './data/driving_log.csv'
DEFAULT_MODEL_PATH = 'model.h5'


def parse_log(filepath=DEFAULT_LOG_PATH):
    """Parse driving log file"""
    lines = []

    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            lines.append(line)

    return lines


def extract_data(filepath=DEFAULT_LOG_PATH, header=True):
    """Extract images and measurements from driving logs"""
    images = []
    measurements = []
    lines = parse_log(filepath)

    if header:
        lines = lines[1:]

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]

        image = cv2.imread('data/IMG/{}'.format(filename))
        images.append(image)

        measurement = float(line[3])
        measurements.append(measurement)

    return np.array(images), np.array(measurements)


# Models


def build_model():
    """Build model to training"""
    model = Sequential()

    # Normalization
    # model.add()
    # model.add(
    #     Lambda(lambda x: x / 255.0 - 0.5,
    #            input_shape=(162, 320, 3)))
    model.add(Flatten(input_shape=(162, 320, 3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model


# Training

def fit_model(model, X_train, y_train, save_path=DEFAULT_MODEL_PATH):
    """Fit model to data."""
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        shuffle=True
    )
    if save_path:
        model.save('model.h5')


# Load data and train model
if __name__ == '__main__':
    model = build_model()
    X_train, y_train = extract_data()

    if sys.argv:
        save_path = sys.argv[0]
    else:
        save_path = DEFAULT_MODEL_PATH

    fit_model(model, X_train, y_train, save_path=save_path)
