import tensorflow as tf

from mink import inits
from mink import nonlinearities
from mink.utils import as_4d
from mink.utils import as_tuple
from mink.utils import get_shape

from .base import Layer


__all__ = ['Conv2DLayer']


# pylint: disable=too-many-instance-attributes,too-many-arguments
class Conv2DLayer(Layer):
    """TODO"""
    def __init__(
            self,
            incoming=None,
            num_filters=32,
            filter_size=3,
            stride=1,
            padding='SAME',
            W=inits.GlorotUniform(),
            b=inits.Constant(0.),
            nonlinearity=nonlinearities.Rectify(),
            name=None,
            make_logs=False,
    ):
        self.incoming = incoming
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.W = W
        self.b = b
        self.nonlinearity = nonlinearity
        self.name = name
        self.make_logs = make_logs

        allowed = ('SAME', 'VALID')
        if padding not in allowed:
            raise ValueError("`padding` must be one of {}.".format(
                ', '.join(allowed)))

    def fit(self, Xs_inc, ys=None, **kwargs):
        filter_size = as_tuple(
            self.filter_size,
            num=2,
            dtype=int,
        )

        self.strides_ = as_4d(self.stride)

        self.add_param('W_', self.W((
            filter_size[0],
            filter_size[1],
            get_shape(Xs_inc)[3],
            self.num_filters,
        )))

        self.add_param('b_', self.b((self.num_filters,)))

        return self

    def transform(self, Xs_inc, **kwargs):
        conved = tf.nn.conv2d(
            Xs_inc,
            filter=self.W_,
            strides=self.strides_,
            padding=self.padding,
        )

        activation = tf.nn.bias_add(conved, self.b_)
        return self.nonlinearity(activation)
