from sklearn.base import BaseEstimator
import tensorflow as tf


class Nonlinearity(BaseEstimator):
    pass


class Rectify(Nonlinearity):
    def __call__(self, Xs, ys=None):
        return tf.nn.relu(Xs)


class Softmax(Nonlinearity):
    def __call__(self, Xs, ys=None):
        return tf.nn.softmax(Xs)
