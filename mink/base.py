from nolearn.lasagne import BatchIterator
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

from mink.config import floatX
from mink.objectives import Objective
from mink.updates import SGD


class NeuralNet(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            layer,
            objective=Objective(),
            update=SGD(),
            batch_iterator=BatchIterator(256),
            max_epochs=10,
            Xs=None,
            ys=None,
            verbose=0,
            binarizer=LabelBinarizer(),
    ):
        self.layer = layer
        self.objective = objective
        self.update = update
        self.batch_iterator = batch_iterator
        self.max_epochs = max_epochs
        self.Xs = Xs
        self.ys = ys
        self.verbose = verbose
        self.binarizer = binarizer

    def _initialize(self, X=None, y=None):
        if getattr(self, '_initalized', None):
            return

        if (X is None) and (self.Xs is None):
            raise ValueError
        if (y is None) and (self.ys is None):
            raise ValueError

        Xs = self.Xs or tf.placeholder(
            dtype=floatX,
            shape=[None] + list(X.shape[1:]),
        )
        ys = self.ys or tf.placeholder(
            dtype=floatX,
            shape=[None] + list(y.shape[1:]),
        )
        ys_proba = self.layer.fit_transform(Xs)

        loss = self.objective(ys, ys_proba)
        train_step = self.update(loss)

        self.loss_ = loss
        self.train_step_ = train_step
        self.Xs_ = Xs
        self.ys_ = ys
        self._predict_proba = ys_proba
        self._initialized = True

    def fit(self, X, yt, num_epochs=None):
        if self.binarizer:
            y = self.binarizer.fit_transform(yt).astype(np.float32)
        else:
            y = yt

        self._initialize(X, y)
        if num_epochs is None:
            num_epochs = self.max_epochs

        session = tf.Session()
        session.run(tf.initialize_all_variables())
        self.session_ = session

        for epoch in range(num_epochs):
            losses = []
            for Xb, yb in self.batch_iterator(X, y):
                feed_dict = {self.Xs_: Xb, self.ys_: yb}
                session.run(self.train_step_, feed_dict=feed_dict)
                if self.verbose:
                    loss = session.run(self.loss_, feed_dict=feed_dict)
                    losses.append(loss)
            if self.verbose:
                print(np.mean(loss))

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
