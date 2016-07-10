from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import tensorflow as tf

from mink.utils import get_layer_name
from mink.utils import get_shape
from mink.utils import set_named_layer_param


__all__ = ['Layer', 'FunctionLayer']
flags = tf.app.flags


class Layer(BaseEstimator, TransformerMixin):
    """TODO"""
    def __init__(self, incoming=None, name=None, make_logs=False):
        raise NotImplementedError

    def initialize(self, Xs, ys=None, **kwargs):
        incomings = getattr(self, 'incoming', 0) or getattr(self, 'incomings')
        if not isinstance(incomings, list):
            incomings = [incomings]

        for incoming in incomings:
            incoming.initialize(Xs, ys, **kwargs)

        Xs_incs = [incoming.get_output(Xs, **kwargs) for incoming in incomings]
        if len(Xs_incs) == 1:
            Xs_incs = Xs_incs[0]

        self.fit(Xs_incs, ys, **kwargs)
        return self

    def fit(self, Xs_inc, ys=None, **kwargs):
        return self

    def get_output(self, Xs, **kwargs):
        # handle transformation of Xs
        incomings = getattr(self, 'incoming', 0) or getattr(self, 'incomings')
        if not isinstance(incomings, list):
            incomings = [incomings]

        Xs_incs = [incoming.get_output(Xs, **kwargs) for incoming in incomings]
        if len(Xs_incs) == 1:
            Xs_incs = Xs_incs[0]

        X_out = self.transform(Xs_incs, **kwargs)
        self.output_shape = get_shape(X_out)

        # handle tensorflow logging
        if hasattr(flags.FLAGS, 'summaries_dir') and self.make_logs:
            layer_name = get_layer_name(self)
            tf.histogram_summary(layer_name + ' activity', X_out)
        return X_out

    def transform(self, Xs_inc, **kwargs):
        raise NotImplementedError

    def add_param(
            self,
            spec,
            name,
            shape=None,
            force=False,
    ):
        if not name.endswith('_'):
            raise ValueError("Parameter names should end in '_', e.g. 'W_'.")

        if not hasattr(self, 'params_'):
            self.params_ = {}

        if (name in self.params_) and not force:
            return

        if hasattr(spec, 'get_shape'):
            if (shape is not None) and (get_shape(spec) != shape):
                raise ValueError("Inconsistent shapes: {} and {}.".format(
                    get_shape(spec), shape))
            param = spec
        elif shape is None:
            raise TypeError('Cannot add this parameter without a shape.')
        else:
            param = spec(shape)

        if not isinstance(param, (tf.Variable, tf.Tensor)):
            param = tf.Variable(param)

        if force or (name not in self.params_):
            self.params_[name] = param
            self.__dict__[name] = param

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
    """TODO"""
    def __init__(self, incoming=None, func=None, name=None, make_logs=False):
        self.incoming = incoming
        self.func = func
        self.name = name
        self.make_logs = make_logs

    def transform(self, Xs_inc, **kwargs):
        func = self.func if self.func is not None else _identity
        return func(Xs_inc)
