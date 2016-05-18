from sklearn.base import BaseEstimator
import tensorflow as tf


def cross_entropy(y_true, y_proba):
    return tf.reduce_mean(-tf.reduce_sum(
        y_true * tf.log(y_proba),
        reduction_indices=[1],
    ))


class Objective(BaseEstimator):
    def __init__(
            self,
            loss_function=cross_entropy,
    ):
        self.loss_function = loss_function

    def __call__(self, y_true, y_proba):
        return self.loss_function(y_true, y_proba)
