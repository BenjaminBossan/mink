import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import tensorflow as tf

from mink import inits
from mink import nonlinearities
from mink.utils import as_tuple
from mink.utils import flatten
from mink.utils import get_shape
from mink.utils import set_named_layer_param


__all__ = [
    'InputLayer',
    'DenseLayer',
    'Conv2DLayer',
]


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


def _identity(X):
    """The identity function.
    """
    return X


class FunctionLayer(Layer):
    def __init__(self, func=None):
        self.func = func

    def fit(self, Xs, ys=None):
        return self

    def transform(self, Xs, ys=None):
        func = self.func if self.func is not None else _identity
        return func(Xs)


class InputLayer(Layer):
    def __init__(
            self,
            Xs=None,
            ys=None,
            name=None,
    ):
        self.Xs = Xs
        self.ys = ys
        self.name = name

    def fit(self, Xs, ys=None):
        if self.Xs is None:
            self.Xs_ = Xs
        else:
            self.Xs_ = self.Xs
        if self.ys is None:
            self.ys_ = ys
        else:
            self.ys_ = self.ys
        return self

    def transform(self, Xs, ys=None):
        return self.Xs_


class DenseLayer(Layer):
    def __init__(
        self,
        incoming=None,
        num_units=100,
        nonlinearity=nonlinearities.Rectify(),
        W=inits.GlorotUniform(),
        b=inits.Constant(0.),
        name=None,
    ):
        self.incoming = incoming
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.W = W
        self.b = b
        self.name = name

    def fit(self, Xs, ys=None):
        Xs_inc = self.incoming.fit_transform(Xs, ys)

        shape = get_shape(Xs_inc)
        self.W_ = self.W((np.prod(shape[1:]), self.num_units))
        self.b_ = self.b((1, self.num_units))

        return self

    def transform(self, Xs, ys=None):
        Xs_inc = self.incoming.transform(Xs)
        if len(Xs_inc.get_shape()) > 2:
            Xs_inc = flatten(Xs_inc, 2)

        X = tf.matmul(Xs_inc, self.W_)
        X += self.b_  # use tf.nn.bias_add?
        return self.nonlinearity(X)


class Conv2DLayer(Layer):
    def __init__(
            self,
            incoming,
            num_filters=32,
            filter_size=3,
            stride=1,
            padding='SAME',
            W=inits.GlorotUniform(),
            b=inits.Constant(0.),
            nonlinearity=nonlinearities.Rectify(),
    ):
        self.incoming = incoming
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.W = W
        self.b = b
        self.nonlinearity = nonlinearity

        allowed = ('SAME', 'VALID')
        if padding not in allowed:
            raise ValueError("`padding` must be one of {}.".format(
                ', '.join(allowed)))

    def fit(self, Xs, ys=None):
        Xs_inc = self.incoming.fit_transform(Xs, ys)

        filter_size = as_tuple(
            self.filter_size,
            N=2,
            t=int,
        )

        self.strides_ = (1, self.stride, self.stride, 1)

        self.W_ = self.W((
            filter_size[0],
            filter_size[1],
            get_shape(Xs_inc)[3],
            self.num_filters,
        ))

        self.b_ = self.b((self.num_filters,))

        return self

    def transform(self, Xs, ys=None):
        Xs_inc = self.incoming.transform(Xs)

        conved = tf.nn.conv2d(
            Xs_inc,
            filter=self.W_,
            strides=self.strides_,
            padding=self.padding,
        )

        activation = tf.nn.bias_add(conved, self.b_)
        return self.nonlinearity(activation)
