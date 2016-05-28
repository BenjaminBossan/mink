from sklearn.base import BaseEstimator
import tensorflow as tf


class Objective(BaseEstimator):
    def __call__(self, y_true, y_proba):
        raise NotImplementedError


class CrossEntropy(Objective):
    def __init__(self, eps=1e-12):
        self.eps = eps

    def __call__(self, y_true, y_proba):
        eps = self.eps
        y_clipped = tf.clip_by_value(y_proba, eps, 1 - eps)
        return tf.reduce_mean(-tf.reduce_sum(
            y_true * tf.log(y_clipped),
            reduction_indices=[1],
        ))
