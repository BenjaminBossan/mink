import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

from mink.config import floatX
from mink.nolearn import BatchIterator
from mink.objectives import Objective
from mink.updates import SGD
from mink.utils import get_input_layers
from mink.utils import set_named_layer_param


__all__ = ['NeuralNetClassifier']


class NeuralNetClassifier(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            layer,
            objective=Objective(),
            update=SGD(),
            batch_iterator=BatchIterator(256),
            max_epochs=10,
            verbose=0,
            binarizer=LabelBinarizer(),
            session=None,
    ):
        self.layer = layer
        self.objective = objective
        self.update = update
        self.batch_iterator = batch_iterator
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.binarizer = binarizer
        self.session = session

    def _initialize(self, X, y):
        if getattr(self, '_initalized', None):
            return

        input_layer = get_input_layers(self.layer)[0]

        Xs = input_layer.Xs or tf.placeholder(
            dtype=floatX,
            shape=[None] + list(X.shape[1:]),
        )
        ys = input_layer.ys or tf.placeholder(
            dtype=floatX,
            shape=[None] + list(y.shape[1:]),
        )
        ys_proba = self.layer.fit_transform(Xs)

        loss = self.objective(ys, ys_proba)
        train_step = self.update(loss)

        if self.session is None:
            self.session_ = tf.Session()
        else:
            self.session_ = self.session
        self.session_.run(tf.initialize_all_variables())

        self.loss_ = loss
        self.train_step_ = train_step
        self.Xs_ = Xs
        self.ys_ = ys
        self._predict_proba = ys_proba
        self._initialized = True

    @property
    def classes_(self):
        return self.binarizer.classes_

    def fit(self, X, yt, num_epochs=None):
        if self.binarizer:
            y = self.binarizer.fit_transform(yt).astype(np.float32)
        else:
            y = yt

        self._initialize(X, y)
        if num_epochs is None:
            num_epochs = self.max_epochs

        for i, epoch in enumerate(range(num_epochs)):
            losses = []
            for Xb, yb in self.batch_iterator(X, y):
                feed_dict = {self.Xs_: Xb, self.ys_: yb}
                __, loss = self.session_.run(
                    [self.train_step_, self.loss_],
                    feed_dict=feed_dict,
                )
                if self.verbose:
                    losses.append(loss)
            if self.verbose:
                # TODO: should use np.average at some point
                print(i + 1, np.mean(loss))

        return self

    def predict_proba(self, X):
        session = self.session_
        y_proba = []

        for Xb, __ in self.batch_iterator(X):
            feed_dict = {self.Xs_: Xb}
            y_proba.append(
                session.run(self._predict_proba, feed_dict=feed_dict))
        return np.vstack(y_proba)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)

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
