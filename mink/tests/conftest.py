import pytest
from sklearn.datasets import make_classification

from mink import NeuralNetClassifier
from mink import layers
from mink import nonlinearities


@pytest.fixture
def clf_data():
    return make_classification(
        n_samples=500,
        n_classes=5,
        n_informative=10,
        random_state=0,
    )


@pytest.fixture
def clf_layers():
    l = layers.InputLayer(name='input')
    l = layers.DenseLayer(l, name='dense')
    l = layers.DenseLayer(
        l,
        num_units=5,
        nonlinearity=nonlinearities.Softmax(),
        name='output',
    )
    return l


@pytest.fixture
def clf_net(clf_layers):
    return NeuralNetClassifier(layer=clf_layers)
