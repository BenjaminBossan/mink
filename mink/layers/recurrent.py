import tensorflow as tf

from .. import nonlinearities
from .base import Layer

__all__ = [
    'RecurrentLayer',
    'LSTMLayer',
    'GRULayer',
]


class RecurrentLayer(Layer):
    """Basic recurrent layer.

    Uses tensorflow's dynamic_rnn.

    Only returns outputs, not cell states.

    Does not (yet) have a good support for masks, as `sequence_length`
    has to be passed during construction of the graph, at which point
    we usually don't yet know the sequence length of every batch.

    """
    def __init__(
            self,
            incoming=None,
            cell=None,
            sequence_length=None,
            name=None,
            make_logs=False,
    ):
        self.incoming = incoming
        self.cell = cell
        self.sequence_length = sequence_length
        self.name = name
        self.make_logs = make_logs

    def fit(self, Xs_inc, ys=None, **kwargs):
        if self.cell is None:
            self.cell_ = tf.nn.rnn_cell.BasicRNNCell(num_units=100)
        else:
            self.cell_ = self.cell

        return self

    def transform(self, Xs_inc, **kwargs):
        return tf.nn.dynamic_rnn(
            inputs=Xs_inc,
            cell=self.cell_,
            dtype=tf.float32,
            sequence_length=self.sequence_length,
        )[0]


class LSTMLayer(RecurrentLayer):
    def __init__(
            self,
            incoming=None,
            num_units=100,
            use_peepholes=False,
            cell_clip=None,
            nonlinearity=nonlinearities.Tanh(),
            sequence_length=None,
            name=None,
            make_logs=False,
    ):
        self.incoming = incoming
        self.num_units = num_units
        self.use_peepholes = use_peepholes
        self.cell_clip = cell_clip
        self.nonlinearity = nonlinearity
        self.sequence_length = sequence_length
        self.name = name
        self.make_logs = make_logs

    def fit(self, Xs_inc, ys=None, **kwargs):
        self.cell_ = tf.nn.rnn_cell.LSTMCell(
            num_units=self.num_units,
            use_peepholes=self.use_peepholes,
            cell_clip=self.cell_clip,
            activation=self.nonlinearity,
            state_is_tuple=True,
        )
        return self


class GRULayer(RecurrentLayer):
    def __init__(
            self,
            incoming=None,
            num_units=100,
            nonlinearity=nonlinearities.Tanh(),
            sequence_length=None,
            name=None,
            make_logs=False,
    ):
        self.incoming = incoming
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.sequence_length = sequence_length
        self.name = name
        self.make_logs = make_logs

    def fit(self, Xs_inc, ys=None, **kwargs):
        self.cell_ = tf.nn.rnn_cell.GRUCell(
            num_units=self.num_units,
            activation=self.nonlinearity,
        )
        return self
