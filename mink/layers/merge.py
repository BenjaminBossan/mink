import tensorflow as tf

from .base import Layer

__all__ = ['ConcatLayer']


class ConcatLayer(Layer):
    """TODO"""
    def __init__(
            self,
            incomings=None,
            axis=1,
            name=None,
            make_logs=False,
    ):
        self.incomings = incomings
        self.axis = axis
        self.name = name
        self.make_logs = make_logs

    def transform(self, Xs_incs, **kwargs):
        return tf.concat(
            values=Xs_incs,
            concat_dim=self.axis,
        )
