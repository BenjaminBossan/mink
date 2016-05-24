from sklearn.base import BaseEstimator
import tensorflow as tf


class Init(BaseEstimator):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class InitNormal(Init):
    def __init__(
            self,
            mean=0.0,
            stddev=1.0,
    ):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape):
        return tf.Variable(tf.random_normal(
            shape=shape,
            mean=self.mean,
            stddev=self.stddev,
        ))


class InitTruncatedNormal(Init):
    def __init__(
            self,
            mean=0.0,
            stddev=1.0,
    ):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape):
        return tf.Variable(tf.truncated_normal(
            shape=shape,
            mean=self.mean,
            stddev=self.stddev,
        ))


class InitZeros(Init):
    def __call__(self, shape):
        return tf.Variable(tf.zeros(
            shape=shape,
        ))
