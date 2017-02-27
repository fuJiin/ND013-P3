from __future__ import absolute_import

import csv

import numpy as np
from sklearn.utils import shuffle

import cv2

DEFAULT_LOG_PATH = './data/driving_log.csv'


def generator(lines, batch_size=128):
    """
    Generator to pass images in batches to model.
    Uses half batch dize because we flip images for each batch.
    """
    normal_lines = [(l, False) for l in lines]
    flipped_lines = [(l, True) for l in lines]

    lines = np.concatenate((normal_lines, flipped_lines), axis=0)
    num_samples = len(lines)

    # Keep generating as long as caller keeps asking
    while True:
        shuffle(lines)

        # Generate batches
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images = []
            angles = []

            # Materialize images from lines
            for line, flipped in batch_samples:
                filepath = source_to_filepath(line[0])

                image = cv2.imread(filepath)
                angle = float(line[3])

                if flipped:
                    image, angle = flip_data(image, angle)

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


def parse_log(filepath=DEFAULT_LOG_PATH):
    """Parse driving log file"""
    lines = []

    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            lines.append(line)

    return lines


def extract_data(filepath=DEFAULT_LOG_PATH, header=False):
    """Extract images and measurements from driving logs in sync"""
    images = []
    measurements = []
    lines = parse_log(filepath)

    if header:
        lines = lines[1:]

    for line in lines:
        filepath = source_to_filepath(line[0])
        image = cv2.imread(filepath)
        images.append(image)

        measurement = float(line[3])
        measurements.append(measurement)

    images, measurements = add_flip_data(images, measurements)
    return images, measurements


def flip_data(x, y):
    return np.fliplr(x), -y


def add_flip_data(X_train, y_train):
    """Flip image and angle data"""
    X_flipped = np.array([np.fliplr(x) for x in X_train])
    y_flipped = np.array([-y for y in y_train])

    X_full = np.concatenate((X_train, X_flipped), axis=0)
    y_full = np.concatenate((y_train, y_flipped), axis=0)

    return X_full, y_full


def source_to_filepath(source_path):
    """Converts source parts to filepath"""
    source_parts = source_path.split('/')
    return './{}'.format('/'.join(source_parts[-3:]))
