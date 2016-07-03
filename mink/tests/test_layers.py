# pylint: disable=invalid-name,missing-docstring,no-self-use
# pylint: disable=old-style-class,no-init

from unittest.mock import Mock

import pytest
import tensorflow as tf

from mink import layers


input_layers = [layers.InputLayer]

layers_2d = [layers.DenseLayer,
             layers.FunctionLayer,
             layers.DenseLayer,
             layers.DropoutLayer]

layers_4d = [layers.Conv2DLayer,
             layers.MaxPool2DLayer,
             layers.ImageResizeLayer]

layers_incs = [layers.ConcatLayer]


class TestAllLayers:
    @pytest.fixture
    def Xs(self):
        return tf.placeholder(dtype=tf.float32, shape=(None, 10))

    @pytest.fixture
    def Xs_4d(self):
        return tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 3))

    @pytest.fixture
    def ys(self):
        return tf.placeholder(dtype=tf.int32, shape=(None, 1))

    @pytest.fixture
    def attributes(self):
        return ['fit', 'transform', 'fit_transform', 'get_output',
                'initialize']

    @pytest.fixture
    def mock_layer_and_mock(self):
        from mink.layers import Layer

        def fit_transform(self, X, y, **kwargs):
            return mock.fit(X, y, **kwargs).transform(X, **kwargs)

        def transform(self, X, **kwargs):
            return X

        mock = Mock()
        mock.fit.return_value = mock
        mock.fit_transform = fit_transform
        mock.transform = transform
        mock.incoming = None

        class MockLayer(Layer):
            fit = mock.fit
            transform = mock.transform
            fit_transform = mock.fit_transform

            def __init__(self, incoming=None):
                self.incoming = incoming

            def initialize(self, Xs, ys, **kwargs):
                self.fit(Xs, ys, **kwargs)
                return self

            def get_output(self, Xs, **kwargs):
                return Xs

        return MockLayer(), mock

    @pytest.fixture
    def input_layer(self, mock_layer_and_mock, Xs, ys):
        mock_layer, mock = mock_layer_and_mock
        mock.transform.return_value = Xs
        mock.fit_transform.return_value = Xs
        mock.Xs = Xs
        mock.ys = ys
        return mock_layer

    @pytest.fixture
    def input_layer_4d(self, mock_layer_and_mock, Xs_4d, ys):
        mock_layer, mock = mock_layer_and_mock
        mock.transform.return_value = Xs_4d
        mock.Xs = Xs_4d
        mock.ys = ys
        return mock_layer

    @pytest.mark.parametrize(
        'layer_cls', input_layers + layers_2d + layers_4d + layers_incs)
    def test_layer_has_required_attributes(self, layer_cls, attributes):
        for attr in attributes:
            assert attr in dir(layer_cls)

    @pytest.mark.parametrize('layer_cls', layers_2d)
    def test_initialize_calls_fit_on_incoming(
            self, layer_cls, input_layer, Xs, ys):
        layer = layer_cls(input_layer)
        layer.initialize(Xs, ys)

        assert input_layer.fit.call_count == 1
        assert input_layer.fit.call_args_list[0][0][0] == Xs
        assert input_layer.fit.call_args_list[0][0][1] == ys

    @pytest.mark.parametrize('layer_cls', layers_4d)
    def test_initialize_calls_fit_on_incoming_4d(
            self, layer_cls, input_layer_4d, Xs_4d, ys):
        layer = layer_cls(input_layer_4d)
        layer.initialize(Xs_4d, ys)

        assert input_layer_4d.fit.call_count == 1
        assert input_layer_4d.fit.call_args_list[0][0][0] == Xs_4d
        assert input_layer_4d.fit.call_args_list[0][0][1] == ys

    @pytest.mark.parametrize('layer_cls', layers_2d)
    def test_get_output_sets_output_shape(
            self, layer_cls, input_layer, Xs, ys):
        layer = layer_cls(input_layer)
        layer.initialize(Xs, ys)

        assert not hasattr(layer, 'output_shape')
        layer.get_output(Xs)
        assert isinstance(layer.output_shape, tuple)

    @pytest.mark.parametrize('layer_cls', layers_4d)
    def test_get_output_sets_output_shape_4d(
            self, layer_cls, input_layer_4d, Xs_4d, ys):
        layer = layer_cls(input_layer_4d)
        layer.initialize(Xs_4d, ys)

        assert not hasattr(layer, 'output_shape')
        layer.get_output(Xs_4d)
        assert isinstance(layer.output_shape, tuple)
