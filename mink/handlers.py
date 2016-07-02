"""Contains handlers for the various callbacks of the NeuralNet
classes.

"""

from collections import OrderedDict
import functools
import operator
import sys

from sklearn.base import BaseEstimator
from tabulate import tabulate

from mink.utils import get_all_layers
from mink.utils import get_layer_name


class ansi:
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class Handler(BaseEstimator):
    def _clear(self):
        pass

    def __call__(self, net):
        raise NotImplementedError


class PrintTrainProgress(object):
    """Print training progress after each epoch."""
    def __init__(
            self,
            tablefmt='pipe',
            floatfmt='.5f',
            first_iteration=True,
    ):
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt
        self.first_iteration = first_iteration

    def _clear(self):
        self.first_iteration = True

    def __call__(self, net):
        if not net.verbose:
            return

        print(self.table(net.train_history_))
        sys.stdout.flush()

    def table(self, history):
        info = history[-1]
        train_loss = info['train loss']
        best_train_loss = train_loss <= min(
            step['train loss'] for step in history)

        info_tabulate = OrderedDict([
            ('epoch', info['epoch']),
            ('train loss', "{}{:.5f}{}".format(
                ansi.CYAN if best_train_loss else "",
                info['train loss'],
                ansi.ENDC if best_train_loss else "",
            )),
            ('dur', info['dur']),
        ])

        tabulated = tabulate(
            [info_tabulate],
            headers="keys",
            tablefmt=self.tablefmt,
            floatfmt=self.floatfmt,
        )

        out = ""
        if self.first_iteration:
            out = "\n".join(tabulated.split('\n', 2)[:2])
            out += "\n"
            self.first_iteration = False
        out += tabulated.rsplit('\n', 1)[-1]

        return out


class PrintLayerInfo(object):
    """Print basic information about the net's layers."""
    def __init__(self, tablefmt='pipe'):
        self.tablefmt = tablefmt

    def __call__(self, net):
        if not net.verbose:
            return

        message = self.get_greeting(net)
        print(message)
        print("## Layer information")
        print("")

        layer_info = self.get_layer_info_plain(net)
        print(layer_info)
        print("")
        sys.stdout.flush()

    @staticmethod
    def get_greeting(net):
        """Information about the number of learnable parameters."""
        all_params = net.get_all_params()
        shapes = []
        for params in all_params:
            for param in params.values():
                shapes.append(param.shape)

        num_params = functools.reduce(
            operator.add,
            [functools.reduce(operator.mul, shape) for shape in shapes])

        message = ("# Neural Network with {} learnable parameters"
                   "\n".format(num_params))
        return message

    def get_layer_info_plain(self, net):
        """Information about the layer output shapes."""
        all_layers = get_all_layers(net.layer)

        nums = list(range(len(all_layers)))
        names = [get_layer_name(layer) for layer in all_layers]
        output_shapes = ['x'.join(map(str, layer.output_shape[1:]))
                         for layer in all_layers]

        table = OrderedDict([
            ('#', nums),
            ('name', names),
            ('size', output_shapes),
        ])
        layer_infos = tabulate(
            table,
            headers='keys',
            tablefmt=self.tablefmt,
        )
        return layer_infos
