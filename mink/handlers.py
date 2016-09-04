"""Contains handlers for the various callbacks of the NeuralNet
classes.

"""

from collections import OrderedDict
import functools
import itertools
import operator
import sys

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.scorer import check_scoring
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


class PrintTrainProgress(Handler):
    """Print training progress after each epoch."""
    def __init__(
            self,
            scores_to_minimize=None,
            scores_to_maximize=None,
            tablefmt='pipe',
            floatfmt='.5f',
            first_iteration=True,
    ):
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt
        self.first_iteration = first_iteration

        self.scores_to_minimize = scores_to_minimize or ['train loss']
        if 'train loss' not in self.scores_to_minimize:
            self.scores_to_minimize.append('train loss')
        self.scores_to_maximize = scores_to_maximize or []

        self.min_scores = {key: np.inf for key in self.scores_to_minimize}
        self.max_scores = {key: -np.inf for key in self.scores_to_maximize}
        self.template = "{}{:" + self.floatfmt + "}{}"

    def _clear(self):
        self.first_iteration = True

    def __call__(self, net):
        if not net.verbose:
            return

        print(self.table(net.train_history_))
        sys.stdout.flush()

    def table(self, history):
        colors = itertools.cycle([
            ansi.CYAN, ansi.GREEN, ansi.MAGENTA, ansi.RED])
        info = history[-1]

        for key in self.scores_to_minimize:
            self.min_scores[key] = min(info[key], self.min_scores[key])

        for key in self.scores_to_maximize:
            self.max_scores[key] = max(info[key], self.max_scores[key])

        table = [("epoch", info['epoch'])]

        for key, val in info.items():
            if key in ['epoch', 'dur']:
                continue

            is_best = None
            if key in self.scores_to_minimize:
                is_best = val == self.min_scores[key]
                color = next(colors)
            elif key in self.scores_to_maximize:
                is_best = val == self.max_scores[key]
                color = next(colors)

            table.append((key, self.template.format(
                color if is_best else "",
                info[key],
                ansi.ENDC if is_best else "",
            )))

        table.append(("dur", int(info['dur'])))

        tabulated = tabulate(
            [OrderedDict(table)],
            headers='keys',
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


class PrintLayerInfo(Handler):
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


class ValidationScoreHandler(Handler):
    def __init__(self, X, y, scoring, scoring_name=None):
        self.X = X
        self.y = y
        self.scoring = scoring
        self.scoring_name = scoring_name

    def __call__(self, net):
        scorer = check_scoring(net, scoring=self.scoring)
        scoring_name = self.scoring_name or str(scorer)
        score = scorer(net, self.X, self.y)

        net.train_history_[-1][scoring_name] = score

    def __repr__(self):
        return "ValidationHandler({}, {})".format(
            self.scoring,
            self.scoring_name,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['X'], state['y']
        return state
