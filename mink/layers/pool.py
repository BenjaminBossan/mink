import tensorflow as tf

from mink.utils import as_4d
from mink.utils import as_tuple
from mink.utils import get_shape

from .base import Layer


__all__ = ['ImageResizeLayer', 'MaxPool2DLayer']


class MaxPool2DLayer(Layer):
    """TODO"""
    def __init__(
            self,
            incoming=None,
            pool_size=2,
            stride=2,
            padding='SAME',
            name=None,
            make_logs=False,
    ):
        self.incoming = incoming
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.name = name
        self.make_logs = make_logs

        allowed = ('SAME', 'VALID')
        if padding not in allowed:
            raise ValueError("`padding` must be one of {}.".format(
                ', '.join(allowed)))

    def fit(self, Xs_inc, ys, **kwargs):
        self.pool_size_ = as_4d(self.pool_size)
        self.strides_ = as_4d(self.stride)
        return self

    def transform(self, Xs_inc, **kwargs):
        return tf.nn.max_pool(
            Xs_inc,
            ksize=self.pool_size_,
            strides=self.strides_,
            padding=self.padding,
        )


class ImageResizeLayer(Layer):
    """TODO"""
    def __init__(
            self,
            incoming=None,
            scale=2,
            resize_method=tf.image.ResizeMethod.BILINEAR,
            name=None,
            make_logs=False,
    ):
        self.incoming = incoming
        self.scale = scale
        self.resize_method = resize_method
        self.name = name
        self.make_logs = make_logs

    def fit(self, Xs_inc, ys, **kwargs):
        _, height, width, _ = get_shape(Xs_inc)
        scale_height, scale_width = as_tuple(self.scale, 2)

        self.new_height_ = int(scale_height * height)
        self.new_width_ = int(scale_width * width)
        return self

    def transform(self, Xs_inc, **kwargs):
        return tf.image.resize_images(
            images=Xs_inc,
            new_height=self.new_height_,
            new_width=self.new_width_,
            method=self.resize_method,
        )
