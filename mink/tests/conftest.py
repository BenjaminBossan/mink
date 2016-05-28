import pytest
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

from mink import NeuralNetClassifier
from mink import NeuralNetRegressor
from mink import layers


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
    X, y = make_regression(
        n_samples=2000,
        n_targets=1,
        n_informative=10,
        random_state=0,
    )
    return X, y.reshape(-1, 1)


@pytest.fixture
def _layers():
    l = layers.InputLayer(name='input')
    l = layers.DenseLayer(l, name='dense', num_units=50)
    l = layers.DenseLayer(l, name='output')
    return l


@pytest.fixture
def clf_net(_layers):
    return NeuralNetClassifier(layer=_layers)


@pytest.fixture
def regr_net(_layers):
    net = NeuralNetRegressor(layer=_layers)
    net.set_params(update__learning_rate=0.001)
    return net
