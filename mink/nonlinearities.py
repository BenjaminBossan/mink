from sklearn.base import BaseEstimator
import tensorflow as tf


__all__ = [
    'Rectify',
    'Softmax',
]


class Nonlinearity(BaseEstimator):
    def __call__(self, Xs):
        raise NotImplementedError


class Rectify(Nonlinearity):
    def __call__(self, Xs):
        return tf.nn.relu(Xs)


class Softmax(Nonlinearity):
    def __call__(self, Xs):
        return tf.nn.softmax(Xs)
