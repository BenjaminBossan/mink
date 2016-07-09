from mink.utils import get_shape

from .base import Layer

__all__ = ['InputLayer']


class InputLayer(Layer):
    """TODO"""
    def __init__(
            self,
            Xs=None,
            ys=None,
            name=None,
            make_logs=False,
    ):
        self.Xs = Xs
        self.ys = ys
        self.name = name
        self.make_logs = make_logs

    def initialize(self, Xs, ys, **kwargs):
        self.fit(Xs, ys, **kwargs)
        return self

    def fit(self, Xs, ys=None, **kwargs):
        if self.Xs is None:
            self.Xs_ = Xs
        else:
            self.Xs_ = self.Xs
        if self.ys is None:
            self.ys_ = ys
        else:
            self.ys_ = self.ys
        self.output_shape = get_shape(self.Xs_)
        return self

    def get_output(self, Xs, **kwargs):
        return self.Xs_

    def transform(self, Xs_inc, **kwargs):
        return Xs_inc
