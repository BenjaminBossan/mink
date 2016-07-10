import numpy as np
import tensorflow as tf

from mink import inits
from mink import nonlinearities
from mink.utils import flatten
from mink.utils import get_shape

from .base import Layer

__all__ = ['DenseLayer']


# pylint: disable=too-many-instance-attributes
class DenseLayer(Layer):
    """TODO"""
    def __init__(
            self,
            incoming=None,
            num_units=None,
            nonlinearity=None,
            W=inits.GlorotUniform(),
            b=inits.Constant(0.),
            name=None,
            make_logs=False,
    ):
        self.incoming = incoming
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.W = W
        self.b = b
        self.name = name
        self.make_logs = make_logs

    def fit(self, Xs_inc, ys=None, **kwargs):
        self.num_units_ = self.num_units or 100
        self.nonlinearity_ = self.nonlinearity or nonlinearities.Rectify()

        shape = get_shape(Xs_inc)
        self.add_param(
            spec=self.W,
            shape=(np.prod(shape[1:]), self.num_units_),
            name='W_',
        )
        self.add_param(
            spec=self.b,
            shape=(1, self.num_units_),
            name='b_',
        )

        return self

    def transform(self, Xs_inc, **kwargs):
        if len(Xs_inc.get_shape()) > 2:
            Xs_inc = flatten(Xs_inc, 2)

        X = tf.matmul(Xs_inc, self.W_)
        X += self.b_  # use tf.nn.bias_add?
        return self.nonlinearity_(X)
