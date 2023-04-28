import tensorflow as tf
import numpy as np


TRANSFORMATIONS = ["arcsinh"]


def get_inverse_tf_transformation(transform):
    if transform is None:
        return lambda x: x

    elif transform == 'arcsinh':
        return tf.math.sinh
    else:
        raise ValueError('Unknown transform passed')


def get_tf_transformation(transform):
    if transform is None:
        return lambda x: x

    elif transform == 'arcsinh':
        return tf.math.asinh
    else:
        raise ValueError('Unknown transform passed')


def apply_transformation(transform, values):
    if transform is None:
        return values
    elif transform == 'arcsinh':
        return np.arcsinh(values)
    else:
        raise ValueError('wrong transform value')
