import tensorflow as tf

from .base import Layer


__all__ = ['DropoutLayer']


class DropoutLayer(Layer):
    """TODO"""
    def __init__(
            self,
            incoming=None,
            p=0.5,
            name=None,
            make_logs=False,
    ):
        self.incoming = incoming
        self.p = p  # pylint: disable=invalid-name
        self.name = name
        self.make_logs = make_logs

    def transform(self, Xs_inc, **kwargs):
        deterministic = kwargs.get(
            'deterministic',
            tf.Variable(False))

        keep_prob = 1.0 - self.p
        return tf.cond(
            deterministic,
            lambda: Xs_inc,
            lambda: tf.nn.dropout(Xs_inc, keep_prob=keep_prob),
        )
