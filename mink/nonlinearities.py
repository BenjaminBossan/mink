"""Contains all nonlinearities.

Often, nonlinearities are simple tensorflow expressions. However, it
could still be useful to wrap them as classes for use cases such as
testing `isinstance(nonlin, Nonlinearity)`.

"""

from sklearn.base import BaseEstimator
import tensorflow as tf


__all__ = [
    'Linear',
    'Rectify',
    'Sigmoid',
    'Softmax',
    'Tanh',
]


class Nonlinearity(BaseEstimator):
    def __call__(self, Xs):
        raise NotImplementedError


class Linear(Nonlinearity):
    def __call__(self, Xs):
        return Xs


class Rectify(Nonlinearity):
    def __call__(self, Xs):
        return tf.nn.relu(Xs)


class Sigmoid(Nonlinearity):
    def __call__(self, Xs):
        return tf.nn.sigmoid(Xs)


class Softmax(Nonlinearity):
    def __call__(self, Xs):
        return tf.nn.softmax(Xs)


class Tanh(Nonlinearity):
    def __call__(self, Xs):
        return tf.nn.tanh(Xs)
