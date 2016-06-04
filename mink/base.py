import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

from mink.config import floatX
from mink.layers import DenseLayer
from mink.nolearn import BatchIterator
from mink import handlers
from mink import nonlinearities
from mink import objectives
from mink.updates import SGD
from mink.utils import get_input_layers
from mink.utils import get_all_layers
from mink.utils import get_shape
from mink.utils import set_named_layer_param


__all__ = ['NeuralNetClassifier', 'NeuralNetRegressor']


class NeuralNetBase(BaseEstimator, TransformerMixin):
    def initialize(self, X, y):
        if getattr(self, '_initalized', None):
            return

        Xs, ys = self._get_Xs_ys(X, y)
        deterministic = tf.placeholder(bool)

        self._initialize_output_layer(self.layer, Xs, ys)

        ys_ff = self.layer.fit_transform(Xs, ys, deterministic=deterministic)
        loss = self.objective(ys, ys_ff)
        train_step = self.update(loss)

        if self.session_kwargs:
            self.session_ = tf.Session(**self.session_kwargs)
        else:
            self.session_ = tf.Session()
        # TODO: Only initialize required variables?
        self.session_.run(tf.initialize_all_variables())

        if self.encoder:
            self.encoder.fit(y)

        self.loss_ = loss
        self.train_step_ = train_step
        self.Xs_ = Xs
        self.ys_ = ys
        self.deterministic_ = deterministic
        self.feed_forward_ = ys_ff
        self._initialized = True

    def _get_Xs_ys(self, X, y):
        raise NotImplementedError

    def fit(self, X, yt, num_epochs=None):
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)

        self.initialize(X, yt)

        if self.encoder:
            y = self.encoder.transform(yt)
            if y.shape[1] == 1:  # binary classification:
                y = np.hstack((1.0 - y, y))
        else:
            y = yt

        if num_epochs is None:
            num_epochs = self.max_epochs

        for callback in self.on_training_started:
            callback(self)

        try:
            self.train_loop(X, y, num_epochs=num_epochs)
        except KeyboardInterrupt:
            pass
        return self

    def train_loop(self, X, y, num_epochs):
        template = "epochs: {:>4} | loss: {:.5f}"

        for i, epoch in enumerate(range(num_epochs)):
            losses = []
            for Xb, yb in self.batch_iterator(X, y):
                inputs = [self.train_step_, self.loss_]
                feed_dict = {
                    self.Xs_: Xb,
                    self.ys_: yb,
                    self.deterministic_: False,
                }

                __, loss = self.session_.run(
                    inputs,
                    feed_dict=feed_dict,
                )

                if self.verbose:
                    losses.append(loss)
            if self.verbose:
                # TODO: should use np.average at some point
                print(template.format(i + 1, np.mean(loss)))

        return self

    def predict(self, X):
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
                is_set = set_named_layer_param(self.layer, key, value)

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

    def get_all_params(self):
        all_layers = get_all_layers(self.layer)
        all_params = []
        for layer in all_layers:
            if not hasattr(layer, 'params_'):
                params = {}
            else:
                params = {key: val.eval(self.session_) for key, val
                          in layer.params_.items()}
            all_params.append(params)
        return all_params

    def set_all_params(self, all_params):
        all_layers = get_all_layers(self.layer)

        if len(all_layers) != len(all_params):
            raise ValueError("Networks don't seem to be the same.")

        for layer, params in zip(all_layers, all_params):
            layer_params = getattr(layer, 'params_', {})
            for key, val in params.items():
                if layer_params:
                    layer.add_param(
                        key,
                        layer.params_[key].assign(val),
                        force=True,
                    )
                    # it appears that you have to eval for the effect
                    # to take place
                    layer.params_[key].eval(session=self.session_)
                else:
                    layer.add_param(key, tf.Variable(val))

    def __getstate__(self):
        state = dict(self.__dict__)
        for key in self.__dict__:
            if key.endswith('_'):
                del state[key]
        all_params = self.get_all_params()
        state['_all_params'] = all_params
        return state

    def __setstate__(self, state):
        all_params = state.pop('_all_params')
        self.__dict__ = state
        self.set_all_params(all_params)


class NeuralNetClassifier(NeuralNetBase):
    def __init__(
            self,
            layer,
            objective=objectives.CrossEntropy(),
            update=SGD(),
            batch_iterator=BatchIterator(256),
            max_epochs=10,
            verbose=0,
            encoder=LabelBinarizer(),
            session_kwargs=None,
            on_training_started=[handlers.PrintLayerInfo()],
    ):
        self.layer = layer
        self.objective = objective
        self.update = update
        self.batch_iterator = batch_iterator
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.encoder = encoder
        self.session_kwargs = session_kwargs
        self.on_training_started = on_training_started

    def _initialize_output_layer(self, layer, Xs, ys):
        if isinstance(layer, DenseLayer):
            ys_shape = get_shape(ys)
            if (layer.num_units is None) and (len(ys_shape) == 2):
                layer.set_params(num_units=ys_shape[1])
            if layer.nonlinearity is None:
                layer.set_params(nonlinearity=nonlinearities.Softmax())

    def _get_Xs_ys(self, X, y):
        input_layer = get_input_layers(self.layer)[0]

        Xs = input_layer.Xs or tf.placeholder(
            dtype=floatX,
            shape=[None] + list(X.shape[1:]),
        )
        ys = input_layer.ys or tf.placeholder(
            dtype=floatX,
            shape=[None] + [len(np.unique(y))]
        )
        return Xs, ys

    @property
    def classes_(self):
        return self.encoder.classes_

    def predict_proba(self, X):
        session = self.session_
        y_proba = []

        for Xb, __ in self.batch_iterator(X):
            feed_dict = {self.Xs_: Xb, self.deterministic_: True}
            y_proba.append(
                session.run(self.feed_forward_, feed_dict=feed_dict))
        return np.vstack(y_proba)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)


class NeuralNetRegressor(NeuralNetBase):
    def __init__(
            self,
            layer,
            objective=objectives.MeanSquaredError(),
            update=SGD(),
            batch_iterator=BatchIterator(256),
            max_epochs=10,
            verbose=0,
            encoder=None,
            session_kwargs=None,
            on_training_started=[handlers.PrintLayerInfo()],
    ):
        self.layer = layer
        self.objective = objective
        self.update = update
        self.batch_iterator = batch_iterator
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.encoder = encoder
        self.session_kwargs = session_kwargs
        self.on_training_started = on_training_started

    def _initialize_output_layer(self, layer, Xs, ys):
        if isinstance(layer, DenseLayer):
            ys_shape = get_shape(ys)
            if (layer.num_units is None) and (len(ys_shape) == 2):
                if ys_shape[1] > 1:
                    raise ValueError("Multioutput regression currently not "
                                     "supported.")
                layer.set_params(num_units=ys_shape[1])
            if layer.nonlinearity is None:
                layer.set_params(nonlinearity=nonlinearities.Linear())

    def _get_Xs_ys(self, X, y):
        input_layer = get_input_layers(self.layer)[0]

        Xs = input_layer.Xs or tf.placeholder(
            dtype=floatX,
            shape=[None] + list(X.shape[1:]),
        )
        ys = input_layer.ys or tf.placeholder(
            dtype=floatX,
            shape=[None, 1]  # TODO: Multioutput not supported yet
        )
        return Xs, ys

    def predict(self, X):
        session = self.session_
        y_pred = []

        for Xb, __ in self.batch_iterator(X):
            feed_dict = {self.Xs_: Xb, self.deterministic_: True}
            y_pred.append(
                session.run(self.feed_forward_, feed_dict=feed_dict))
        return np.vstack(y_pred)
