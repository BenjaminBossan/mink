"""Contains all objectives (loss or error functions).

Often, objectives are simple tensorflow expressions. However, it
could still be useful to wrap them as classes for use cases such as
testing `isinstance(obj, Objective)`.

"""

from sklearn.base import BaseEstimator
import tensorflow as tf


class Objective(BaseEstimator):
    def __call__(self, y_true, y_transformed):
        raise NotImplementedError


class CrossEntropy(Objective):
    """Cross entropy error."""
    def __init__(self, eps=1e-12):
        self.eps = eps

    def __call__(self, y_true, y_transformed):
        eps = self.eps
        y_clipped = tf.clip_by_value(y_transformed, eps, 1 - eps)
        return tf.reduce_mean(-tf.reduce_sum(
            y_true * tf.log(y_clipped),
            reduction_indices=[1],
        ))


class MeanSquaredError(Objective):
    """Mean squared error."""
    def __call__(self, y_true, y_transformed):
        return tf.reduce_mean(tf.square(y_true - y_transformed))
