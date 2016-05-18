import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import tensorflow as tf

from mink.utils import get_shape
from mink.inits import InitNormal
from mink.nonlinearities import Rectify


class Layer(BaseEstimator, TransformerMixin):
    def fit(self, Xs, ys=None):
        return self


class InputLayer(Layer):
    def __init__(self, Xs=None):
        self.Xs = Xs

    def fit(self, Xs, ys=None):
        if self.Xs is None:
            self.Xs_ = Xs
        else:
            self.Xs_ = self.Xs
        return self

    def transform(self, Xs, ys=None):
        return self.Xs_


class DenseLayer(Layer):
    def __init__(
        self,
        incoming=None,
        num_units=100,
        nonlinearity=Rectify(),
        W_init=InitNormal(),
        b_init=InitNormal(),
    ):
        self.incoming = incoming
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.W_init = W_init
        self.b_init = b_init

    def fit(self, Xs, ys=None):
        Xs_inc = self.incoming.fit_transform(Xs, ys)

        shape = get_shape(Xs_inc)
        self.W_ = self.W_init((np.prod(shape[1:]), self.num_units))
        self.b_ = self.b_init((1, self.num_units))

        return self

    def transform(self, Xs, ys=None):
        Xs_inc = self.incoming.transform(Xs)
        X = tf.matmul(Xs_inc, self.W_)
        X += self.b_  # use tf.nn.bias_add?
        return self.nonlinearity(X)
