"""Module contains estimators and the estimator base class."""

from collections import defaultdict
import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

from mink import handlers
from mink import nonlinearities
from mink import objectives
from mink.config import floatX
from mink.iterators import IteratorPipeline
from mink.layers import DenseLayer
from mink.updates import SGD
from mink.utils import get_all_layers
from mink.utils import get_input_layers
from mink.utils import get_layer_name
from mink.utils import set_named_layer_param

flags = tf.app.flags


__all__ = ['make_network', 'NeuralNetClassifier', 'NeuralNetRegressor']


def _name_layers(layer_lst):
    """Generate names for layers.

    Only works for linear graph.

    """

    names = [get_layer_name(layer) for layer in layer_lst]
    namecount = defaultdict(int)
    for layer, name in zip(layer_lst, names):
        namecount[name] += 1

    for key, val in list(namecount.items()):
        if val == 1:
            del namecount[key]

    for i, layer in reversed(list(enumerate(layer_lst))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1
        if hasattr(layer, 'incoming'):
            layer.set_params(incoming=layer_lst[i - 1])

    return list(zip(names, layer_lst))


def make_network(layers_lst):
    return Pipeline(_name_layers(layers_lst))


# pylint: disable=super-init-not-called,too-many-arguments
# pylint: disable=too-many-instance-attributes
class NeuralNetBase(BaseEstimator, TransformerMixin):
    """Base class for neural network estimators.

    Cannot be used by itself.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
            self,
            layer,
            update,
            batch_iterator_train,
            batch_iterator_test,
            max_epochs,
            objective,
            encoder,
            session_kwargs,
            on_training_started,
            on_epoch_finished,
            verbose,
    ):
        self.layer = layer
        self.update = update
        self.batch_iterator_train = batch_iterator_train
        self.batch_iterator_test = batch_iterator_test
        self.max_epochs = max_epochs
        self.objective = objective
        self.encoder = encoder
        self.session_kwargs = session_kwargs
        self.on_training_started = on_training_started
        self.on_epoch_finished = on_epoch_finished
        self.verbose = verbose

    def initialize(self, X=None, y=None):
        """TODO"""
        # Note: if X and y are None, we assume that the net has been
        # loaded from a fitted net. In that case, the shapes of inputs
        # and outputs are saved as attributes on the input and output
        # layers and we don't need to infer them from data.

        if getattr(self, '_initialized', None):
            return

        input_shapes = self._get_input_shapes(X)
        if y is None:
            # Assumes that net is initialiazed already
            try:
                output_shape = self.layer.output_shape
            except AttributeError:
                raise AttributeError(
                    "Please initialize the net with data, "
                    "e.g. net.initialiize(X, y).")
        else:
            output_shape = self._get_output_shape(y)

        Xs, ys = self._get_symbolic_vars(input_shapes, output_shape)
        deterministic = tf.placeholder(bool)

        if isinstance(self.layer, list):
            layer = self.layer[-1]
        else:
            layer = self.layer

        self._initialize_output_layer(layer, output_shape)
        layer.initialize(Xs, ys, deterministic=deterministic)
        ys_ff = layer.get_output(Xs, deterministic=deterministic)
        loss = self.objective(ys, ys_ff)
        train_step = self.update(loss)

        if self.session_kwargs:
            session = tf.Session(**self.session_kwargs)
        else:
            session = tf.Session()
        # TODO: Only initialize required variables?
        session.run(tf.initialize_all_variables())

        if hasattr(flags.FLAGS, 'summaries_dir'):
            tensorboard_logs = tf.train.SummaryWriter(
                flags.FLAGS.summaries_dir,
                session.graph,
            )
        else:
            tensorboard_logs = None

        self.batch_iterator_train_, self.batch_iterator_test_ = (
            self._get_iterators())

        if tensorboard_logs:
            tf.histogram_summary('train activity', ys_ff)
            tf.scalar_summary('train loss', loss)

        if y is not None and self.encoder:
            self.encoder.fit(y)

        self.session_ = session
        self.tensorboard_logs_ = tensorboard_logs
        self.loss_ = loss
        self.train_step_ = train_step
        self.Xs_ = Xs
        self.ys_ = ys
        self.deterministic_ = deterministic
        self.feed_forward_ = ys_ff
        self.train_history_ = []
        self._initialized = True

    def _initialize_output_layer(self, layer, output_shape):
        raise NotImplementedError

    def _get_input_shapes(self, X):
        if X is None:
            # Assumes that net is initialiazed already
            try:
                input_layers = get_input_layers(self.layer)
                input_shapes = [l.output_shape for l in input_layers]
            except AttributeError:
                raise AttributeError(
                    "Please initialize the net with data, "
                    "e.g. net.initialiize(X, y).")
        else:
            if isinstance(X, np.ndarray):
                X = [X]
            input_shapes = [[None] + list(x.shape[1:]) for x in X]

        return input_shapes

    def _get_output_shape(self, y):
        # Depends on whether we have classification or regression (or
        # something else entirely).
        raise NotImplementedError

    def _get_symbolic_vars(self, input_shapes, output_shape):
        input_layers = get_input_layers(self.layer)
        if (len(input_shapes) > 1) or (len(input_layers) > 1):
            raise ValueError("Multiple input layers not supported yet.")
        input_layer = input_layers[0]

        Xs = input_layer.Xs if input_layer.Xs is not None else tf.placeholder(
            dtype=floatX,
            shape=input_shapes[0],
        )
        ys = input_layer.ys if input_layer.ys is not None else tf.placeholder(
            dtype=floatX,
            shape=output_shape,
        )
        return Xs, ys

    def _get_iterators(self):
        if isinstance(self.batch_iterator_train, int):
            batch_iterator_train = IteratorPipeline(
                batch_size=self.batch_iterator_train,
                deterministic=False,
            )
        else:
            batch_iterator_train = self.batch_iterator_train

        if isinstance(self.batch_iterator_test, int):
            batch_iterator_test = IteratorPipeline(
                batch_size=self.batch_iterator_test,
                deterministic=True,
            )
        else:
            batch_iterator_test = self.batch_iterator_test
        return batch_iterator_train, batch_iterator_test

    def fit(self, X, yt, epochs=None):
        """TODO"""
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)

        self.initialize(X, yt)

        if self.encoder:
            y = self.encoder.transform(yt)
            if y.shape[1] == 1:  # binary classification:
                y = np.hstack((1.0 - y, y))
        else:
            y = yt

        self.batch_iterator_train_.fit(X, y)
        self.batch_iterator_test_.fit(X, y)

        if epochs is None:
            epochs = self.max_epochs

        for callback in self.on_training_started:
            callback(self)

        try:
            self.train_loop(X, y, epochs=epochs)
        except KeyboardInterrupt:
            pass
        return self

    def train_loop(self, X, y, epochs):
        """TODO"""
        summary = tf.merge_all_summaries()
        inputs = [self.train_step_, self.loss_]
        state = {}
        if summary is not None:
            inputs += [summary]

        for epoch in range(epochs):
            train_losses = []
            tic = time.time()
            state['epoch'] = epoch
            state['train_losses'] = train_losses
            state['tic'] = tic

            for Xb, yb in self.batch_iterator_train_(X, y):
                feed_dict = {
                    self.Xs_: Xb,
                    self.ys_: yb,
                    self.deterministic_: False,
                }

                output = self.session_.run(
                    inputs,
                    feed_dict=feed_dict,
                )
                if summary is not None:
                    _, loss, logs = output
                else:
                    _, loss = output
                    logs = None

                train_losses.append(loss)

            if logs:
                self.tensorboard_logs_.add_summary(logs, epoch)
            if self.verbose:
                self._callback_on_epoch_finished(state)

        return self

    def _callback_on_epoch_finished(self, state):
        info = {
            'epoch': state['epoch'] + 1,
            # TODO: should use np.average at some point
            'train loss': np.mean(state['train_losses']),
            'dur': time.time() - state['tic'],
        }
        self.train_history_.append(info)
        for func in self.on_epoch_finished:
            func(self)

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
        """TODO"""
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
        """TODO"""
        all_layers = get_all_layers(self.layer)

        if len(all_layers) != len(all_params):
            raise ValueError("Networks don't seem to be the same.")

        for layer, params in zip(all_layers, all_params):
            layer_params = getattr(layer, 'params_', {})
            for key, val in params.items():
                if layer_params:
                    layer.add_param(
                        spec=layer.params_[key].assign(val),
                        name=key,
                        force=True,
                    )
                    # it appears that you have to eval for the effect
                    # to take place
                    layer.params_[key].eval(session=self.session_)
                else:
                    layer.add_param(
                        spec=tf.Variable(val),
                        name=key,
                    )

    def __getstate__(self):
        state = dict(self.__dict__)
        for key in self.__dict__:
            if key.endswith('_') or (key == '_initialized'):
                del state[key]
        all_params = self.get_all_params()
        state['_all_params'] = all_params
        return state

    def __setstate__(self, state):
        all_params = state.pop('_all_params')
        self.__dict__ = state
        self.set_all_params(all_params)
        self.initialize()


# pylint: disable=super-init-not-called,too-many-arguments
# pylint: disable=too-many-instance-attributes
class NeuralNetClassifier(NeuralNetBase):
    """TODO"""
    def __init__(
            self,
            layer,
            objective=objectives.CrossEntropy(),
            update=SGD(),
            batch_iterator_train=128,
            batch_iterator_test=128,
            max_epochs=10,
            verbose=0,
            encoder=LabelBinarizer(),
            session_kwargs=None,
            on_training_started=(handlers.PrintLayerInfo(),),
            on_epoch_finished=(handlers.PrintTrainProgress(),),
    ):
        self.layer = layer
        self.objective = objective
        self.update = update
        self.batch_iterator_train = batch_iterator_train
        self.batch_iterator_test = batch_iterator_test
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.encoder = encoder
        self.session_kwargs = session_kwargs
        self.on_training_started = on_training_started
        self.on_epoch_finished = on_epoch_finished

    def _initialize_output_layer(self, layer, output_shape):
        if isinstance(layer, DenseLayer):
            if (layer.num_units is None) and (len(output_shape) == 2):
                layer.set_params(num_units=output_shape[1])
            if layer.nonlinearity is None:
                layer.set_params(nonlinearity=nonlinearities.Softmax())

    def _get_output_shape(self, y):
        num_classes = len(np.unique(y))
        return [None, num_classes]

    @property
    def classes_(self):
        return self.encoder.classes_

    def predict_proba(self, X):
        """TODO"""
        session = self.session_
        y_proba = []

        for Xb, _ in self.batch_iterator_test_(X):
            feed_dict = {self.Xs_: Xb, self.deterministic_: True}
            y_proba.append(
                session.run(self.feed_forward_, feed_dict=feed_dict))
        return np.vstack(y_proba)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)


# pylint: disable=super-init-not-called,too-many-arguments
# pylint: disable=too-many-instance-attributes
class NeuralNetRegressor(NeuralNetBase):
    """TODO"""
    def __init__(
            self,
            layer,
            objective=objectives.MeanSquaredError(),
            update=SGD(),
            batch_iterator_train=128,
            batch_iterator_test=128,
            max_epochs=10,
            verbose=0,
            encoder=None,
            session_kwargs=None,
            on_training_started=(handlers.PrintLayerInfo(),),
            on_epoch_finished=(handlers.PrintTrainProgress(),),
    ):
        self.layer = layer
        self.objective = objective
        self.update = update
        self.batch_iterator_train = batch_iterator_train
        self.batch_iterator_test = batch_iterator_test
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.encoder = encoder
        self.session_kwargs = session_kwargs
        self.on_training_started = on_training_started
        self.on_epoch_finished = on_epoch_finished

    def _initialize_output_layer(self, layer, output_shape):
        if isinstance(layer, DenseLayer):
            if (layer.num_units is None) and (len(output_shape) == 2):
                if output_shape[1] > 1:
                    raise ValueError("Multioutput regression currently not "
                                     "supported.")
                layer.set_params(num_units=output_shape[1])
            if layer.nonlinearity is None:
                layer.set_params(nonlinearity=nonlinearities.Linear())

    def _get_output_shape(self, y):
        output_dim = list(y.shape[1:])
        return [None] + output_dim

    def predict(self, X):
        session = self.session_
        y_pred = []

        for Xb, _ in self.batch_iterator_test_(X):
            feed_dict = {self.Xs_: Xb, self.deterministic_: True}
            y_pred.append(
                session.run(self.feed_forward_, feed_dict=feed_dict))
        return np.vstack(y_pred)
