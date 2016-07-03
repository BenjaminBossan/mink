"""conftest file for pytest.

Contains fixtures and pytest options.

"""

import pytest
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
import tensorflow as tf

from mink import NeuralNetClassifier
from mink import NeuralNetRegressor
from mink import layers


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="run slow tests")


@pytest.fixture
def session_kwargs():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    return {'config': tf.ConfigProto(gpu_options=gpu_options)}


@pytest.fixture
def clf_data():
    return make_classification(
        n_samples=2000,
        n_classes=5,
        n_informative=10,
        random_state=0,
    )


@pytest.fixture
def regr_data():
    return make_regression(
        n_samples=2000,
        n_targets=1,
        n_informative=10,
        random_state=0,
    )


@pytest.fixture
def _layers():
    l = layers.InputLayer(name='input')
    l = layers.DenseLayer(l, name='dense', num_units=50)
    l = layers.DenseLayer(l, name='output')
    return l


@pytest.fixture
def clf_net(_layers, session_kwargs):
    return NeuralNetClassifier(layer=_layers, session_kwargs=session_kwargs)


@pytest.fixture
def regr_net(_layers, session_kwargs):
    net = NeuralNetRegressor(layer=_layers, session_kwargs=session_kwargs)
    net.set_params(update__learning_rate=0.001)
    return net
