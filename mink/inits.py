import numpy as np
from sklearn.base import BaseEstimator
import tensorflow as tf


__all__ = [
    'Constant',
    'Uniform',
    'Normal',
    'TruncatedNormal',
    'GlorotNormal',
    'GlorotUniform',
]


class Init(BaseEstimator):
    def __call__(self, shape):
        raise NotImplementedError


class Constant(Init):
    def __init__(
            self,
            val=0.0,
    ):
        self.val = val

    def __call__(self, shape):
        return tf.Variable(self.val * tf.ones(shape=shape))


class Uniform(Init):
    def __init__(
            self,
            range=0.01,
            std=None,
            mean=0.0,
            seed=None,
    ):
        self.range = range
        self.std = std
        self.mean = mean
        self.seed = seed

    def __call__(self, shape):
        range, std, mean = self.range, self.std, self.mean
        if std is not None:
            high = mean - np.sqrt(3) * std
            low = mean + np.sqrt(3) * std
        else:
            try:
                high, low = range  # range is a tuple
            except TypeError:
                high, low = -range, range  # range is a number

        return tf.Variable(tf.random_uniform(
            shape=shape,
            minval=low,
            maxval=high,
            seed=self.seed,
        ))


class Normal(Init):
    def __init__(
            self,
            std=1.0,
            mean=0.0,
            seed=None,
    ):
        self.std = std
        self.mean = mean
        self.seed = seed

    def __call__(self, shape):
        return tf.Variable(tf.random_normal(
            shape=shape,
            stddev=self.std,
            mean=self.mean,
            seed=self.seed,
        ))


class TruncatedNormal(Init):
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


class Zeros(Init):
    def __call__(self, shape):
        return tf.Variable(tf.zeros(
            shape=shape,
        ))


class Glorot(Init):
    def __init__(
            self,
            initializer,
            gain=np.sqrt(2),
            c01b=False,
    ):
        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b

    def __call__(self, shape):
        if self.c01b:
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            n1, n2 = shape[0], shape[3]
            receptive_field_size = shape[1] * shape[2]
        else:
            if len(shape) < 2:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

            n1, n2 = shape[:2]
            receptive_field_size = np.prod(shape[2:])

        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        return self.initializer(std=std)(shape)


class GlorotNormal(Glorot):
    """Glorot with weights sampled from the Normal distribution.

    See :class:`Glorot` for a description of the parameters.
    """
    def __init__(self, gain=1.0, c01b=False):
        super().__init__(Normal, gain, c01b)


class GlorotUniform(Glorot):
    """Glorot with weights sampled from the Uniform distribution.

    See :class:`Glorot` for a description of the parameters.
    """
    def __init__(self, gain=1.0, c01b=False):
        super().__init__(Uniform, gain, c01b)
