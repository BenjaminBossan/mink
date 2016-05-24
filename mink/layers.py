import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import tensorflow as tf

from mink.utils import get_shape
from mink.inits import InitNormal
from mink.nonlinearities import Rectify
from mink.utils import set_named_layer_param


class Layer(BaseEstimator, TransformerMixin):
    def __init__(self, name=None):
        self.name = name

    def fit(self, Xs, ys=None):
        raise NotImplementedError

    def transform(self, Xs):
        raise NotImplementedError

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested
        objects (such as pipelines). The former have parameters of the
        form ``<component>__<parameter>`` so that it's possible to
        update each component of a nested object.

        Returns
        -------

        self

        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self

        error_msg = ('Invalid parameter {} for estimator {}. '
                     'Check the list of available parameters '
                     'with `estimator.get_params().keys()`.')

        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case

                # try if named layer
                is_set = set_named_layer_param(self, key, value)

                if not is_set:
                    # there was no fitting named layer
                    name, sub_name = split
                    if name not in valid_params:
                        raise ValueError(error_msg.format(name, self))

                    sub_object = valid_params[name]
                    sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError(
                        error_msg.format(key, self.__class__.__name__))
                setattr(self, key, value)
        return self


class InputLayer(Layer):
    def __init__(self, Xs=None, name=None):
        self.Xs = Xs
        self.name = name

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
        name=None,
    ):
        self.incoming = incoming
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.W_init = W_init
        self.b_init = b_init
        self.name = name

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
