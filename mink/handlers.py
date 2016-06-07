from collections import OrderedDict
import functools
import operator
import sys

from tabulate import tabulate

from mink.utils import get_all_layers
from mink.utils import get_layer_name


class PrintLayerInfo(object):
    def __init__(self, tablefmt='pipe'):
        self.tablefmt = tablefmt

    def __call__(self, net):
        if not net.verbose:
            return

        message = self._get_greeting(net)
        print(message)
        print("## Layer information")
        print("")

        layer_info = self._get_layer_info_plain(net)
        print(layer_info)
        print("")
        sys.stdout.flush()

    @staticmethod
    def _get_greeting(net):
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

    def _get_layer_info_plain(self, net):
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
        layer_infos = tabulate(table, 'keys', tablefmt=self.tablefmt)
        return layer_infos
