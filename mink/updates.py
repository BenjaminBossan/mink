from sklearn.base import BaseEstimator
import tensorflow as tf


class Update(BaseEstimator):
    def __call__(self, Xs):
        raise NotImplementedError


class SGD(Update):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def __call__(self, loss):
        train_step = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(loss)
        return train_step


class Momentum(Update):
    def __init__(
            self,
            learning_rate=0.01,
            momentum=0.9,
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def __call__(self, loss):
        train_step = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
        ).minimize(loss)
        return train_step


class Adam(Update):
    def __init__(self, learning_rate=1e-4):
        self.learning_rate = learning_rate

    def __call__(self, loss):
        train_step = tf.train.AdadeltaOptimizer(
            learning_rate=self.learning_rate,
        ).minimize(loss)
        return train_step
