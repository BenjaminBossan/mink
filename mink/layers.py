import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import tensorflow as tf

from mink import inits
from mink import nonlinearities
from mink.utils import as_4d
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

    def fit(self, Xs, ys, **kwargs):
        raise NotImplementedError

    def transform(self, Xs, ys=None, **kwargs):
        raise NotImplementedError

    def add_param(self, name, value, force=False):
        if not hasattr(self, 'params_'):
            self.params_ = {}

        if force or (name not in self.params_):
            self.params_[name] = value
            self.__dict__[name] = value

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

    def __getstate__(self):
        state = dict(self.__dict__)
        for key in self.__dict__:
            if key.endswith('_'):
                del state[key]
        return state


def _identity(X):
    """The identity function.
    """
    return X


class FunctionLayer(Layer):
    def __init__(self, incoming, func=None):
        self.incoming = incoming
        self.func = func

    def fit(self, Xs, ys=None, **kwargs):
        self.incoming.fit(Xs, ys, **kwargs)
        return self

    def transform(self, Xs, ys=None, **kwargs):
        Xs_inc = self.incoming.transform(Xs, ys, **kwargs)
        func = self.func if self.func is not None else _identity
        return func(Xs_inc)


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

    def fit(self, Xs, ys=None, **kwargs):
        if self.Xs is None:
            self.Xs_ = Xs
        else:
            self.Xs_ = self.Xs
        if self.ys is None:
            self.ys_ = ys
        else:
            self.ys_ = self.ys
        return self

    def transform(self, Xs, ys=None, **kwargs):
        return self.Xs_


class DenseLayer(Layer):
    def __init__(
        self,
        incoming=None,
        num_units=None,
        nonlinearity=None,
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

    def fit(self, Xs, ys=None, **kwargs):
        self.num_units_ = self.num_units or 100
        self.nonlinearity_ = self.nonlinearity or nonlinearities.Rectify()

        Xs_inc = self.incoming.fit_transform(Xs, ys, **kwargs)

        shape = get_shape(Xs_inc)
        self.add_param('W_', self.W((np.prod(shape[1:]), self.num_units_)))
        self.add_param('b_', self.b((1, self.num_units_)))

        return self

    def transform(self, Xs, ys=None, **kwargs):
        Xs_inc = self.incoming.transform(Xs, ys, **kwargs)
        if len(Xs_inc.get_shape()) > 2:
            Xs_inc = flatten(Xs_inc, 2)

        X = tf.matmul(Xs_inc, self.W_)
        X += self.b_  # use tf.nn.bias_add?
        return self.nonlinearity_(X)


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
            name=None,
    ):
        self.incoming = incoming
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.W = W
        self.b = b
        self.nonlinearity = nonlinearity
        self.name = name

        allowed = ('SAME', 'VALID')
        if padding not in allowed:
            raise ValueError("`padding` must be one of {}.".format(
                ', '.join(allowed)))

    def fit(self, Xs, ys=None, **kwargs):
        Xs_inc = self.incoming.fit_transform(Xs, ys, **kwargs)

        filter_size = as_tuple(
            self.filter_size,
            N=2,
            t=int,
        )

        self.strides_ = as_4d(self.stride)

        self.add_param('W_', self.W((
            filter_size[0],
            filter_size[1],
            get_shape(Xs_inc)[3],
            self.num_filters,
        )))

        self.add_param('b_', self.b((self.num_filters,)))

        return self

    def transform(self, Xs, ys=None, **kwargs):
        Xs_inc = self.incoming.transform(Xs, ys, **kwargs)

        conved = tf.nn.conv2d(
            Xs_inc,
            filter=self.W_,
            strides=self.strides_,
            padding=self.padding,
        )

        activation = tf.nn.bias_add(conved, self.b_)
        return self.nonlinearity(activation)


class MaxPool2DLayer(Layer):
    def __init__(
            self,
            incoming,
            pool_size=2,
            stride=1,
            padding='SAME',
            name=None,
    ):
        self.incoming = incoming
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.name = name

        allowed = ('SAME', 'VALID')
        if padding not in allowed:
            raise ValueError("`padding` must be one of {}.".format(
                ', '.join(allowed)))

    def fit(self, Xs, ys, **kwargs):
        self.incoming.fit(Xs, ys, **kwargs)

        self.pool_size_ = as_4d(self.pool_size)
        self.strides_ = as_4d(self.stride)

        return self

    def transform(self, Xs, ys=None, **kwargs):
        Xs_inc = self.incoming.transform(Xs, ys, **kwargs)
        return tf.nn.max_pool(
            Xs_inc,
            ksize=self.pool_size_,
            strides=self.strides_,
            padding=self.padding,
        )


class DropoutLayer(Layer):
    def __init__(
            self,
            incoming,
            p=0.5,
    ):
        self.incoming = incoming
        self.p = p

    def fit(self, Xs, ys, **kwargs):
        self.incoming.fit(Xs, ys, **kwargs)
        return self

    def transform(self, Xs, ys=None, **kwargs):
        Xs_inc = self.incoming.transform(Xs, **kwargs)

        deterministic = kwargs.get(
            'deterministic',
            tf.Variable(False))

        keep_prob = 1.0 - self.p
        return tf.cond(
            deterministic,
            lambda: Xs_inc,
            lambda: tf.nn.dropout(Xs_inc, keep_prob=keep_prob),
        )
