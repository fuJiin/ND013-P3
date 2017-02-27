from __future__ import absolute_import


from keras.applications.inception_v3 import InceptionV3
from keras.backend import set_image_dim_ordering
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2

INPUT_SHAPE = (160, 320, 3)

set_image_dim_ordering('tf')


# Models
def _normalize(x):
    return (x / 255.0) - 0.5


def _resize(x):
    import tensorflow as tf
    return tf.image.resize_images(x, (66, 200))


def _crop(x):
    return x[0:, 50:140]


def build_model():
    """Build model to train"""
    model = Sequential()

    # Normalization
    model.add(
        Lambda(lambda x: x / 255.0 - 0.5,
               input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model


def build_model_lenet(reg_beta=5e-4, dropout=0.):
    """Build LeNet"""
    model = Sequential()

    model.add(Lambda(_crop,
                     input_shape=INPUT_SHAPE,
                     name='crop'))
    model.add(Lambda(_normalize,
                     name='normalize'))
    model.add(Convolution2D(6, 5, 5,
                            activation='relu',
                            name='convo_1'))
    model.add(MaxPooling2D(name='max_pool_1'))
    model.add(Convolution2D(6, 5, 5,
                            activation='relu',
                            name='convo_2'))
    model.add(MaxPooling2D(name='max_pool_2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(120, name='dense_1', W_regularizer=l2(reg_beta)))
    model.add(Dense(84, name='dense_2', W_regularizer=l2(reg_beta)))
    model.add(Dense(1, name='dense_3', W_regularizer=l2(reg_beta)))

    if dropout > 0:
        model.add(Dropout(0.5, name='dropout'))

    model.compile(loss='mse', optimizer='adam')
    return model


def build_model_nvidia(reg_beta=5e-4, dropout=0.0):
    """Build Nvidia architecture"""
    model = Sequential()

    model.add(Lambda(_crop,
                     input_shape=INPUT_SHAPE,
                     name='crop'))
    model.add(Lambda(_resize,
                     name='resize'))
    model.add(Lambda(_normalize,
                     name='normalize'))
    model.add(Convolution2D(24, 5, 5,
                            activation='elu',
                            subsample=(2, 2),
                            name='convo_1'))
    model.add(Convolution2D(36, 5, 5,
                            activation='elu',
                            subsample=(2, 2),
                            name='convo_2'))
    model.add(Convolution2D(48, 5, 5,
                            activation='elu',
                            subsample=(2, 2),
                            name='convo_3'))
    model.add(Convolution2D(64, 3, 3,
                            activation='elu',
                            name='convo_4'))
    model.add(Convolution2D(64, 3, 3,
                            activation='elu',
                            name='convo_5'))
    model.add(Flatten(name='flatten'))

    model.add(Dense(100, name='dense_1',
                    # activation='relu',
                    W_regularizer=l2(reg_beta)))
    model.add(Dense(50, name='dense_2',
                    # activation='relu',
                    W_regularizer=l2(reg_beta)))
    model.add(Dense(10, name='dense_3',
                    # activation='relu',
                    W_regularizer=l2(reg_beta)))
    model.add(Dense(1, name='output'))

    if dropout > 0:
        model.add(Dropout(dropout, name='dropout'))

    model.compile(loss='mse', optimizer='adam')
    return model


def build_model_inception():
    """Build and compile Inception"""
    base_model = InceptionV3(
        include_top=False,
        weights=None,
        input_shape=INPUT_SHAPE
    )
    x = base_model.output
    x = Dense(1, name='output')(x)

    model = Model(input=base_model.input, output=x)
    model.compile(loss='mse', optimizer='adam')
    return model
