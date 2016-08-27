# pylint: disable=invalid-name,missing-docstring,no-self-use
# pylint: disable=old-style-class,no-init

from unittest.mock import Mock

import numpy as np
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

        mock = Mock(spec=Layer)
        mock.fit = Mock(return_value=mock)
        mock.transform = Mock(side_effect=lambda X, **kwargs: X)
        mock.fit_transform = fit_transform
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

    def test_call_with_same_input_twice_cached_2d(
            self, mock_layer_and_mock, Xs, ys):
        layer = mock_layer_and_mock[0]
        assert layer.transform._mock_call_count == 0

        layer.fit_transform(Xs, ys)
        assert layer.transform.cache_info().misses == 0
        assert layer.transform.cache_info().hits == 0

        layer.transform(Xs)
        assert layer.transform.cache_info().misses == 1
        assert layer.transform.cache_info().hits == 0

        layer.transform(Xs)
        assert layer.transform.cache_info().misses == 1
        assert layer.transform.cache_info().hits == 1

    def test_call_with_diff_input_twice_not_cached_2d(
            self, mock_layer_and_mock, Xs, ys):
        layer = mock_layer_and_mock[0]
        assert layer.transform._mock_call_count == 0

        layer.fit_transform(Xs, ys)
        assert layer.transform.cache_info().misses == 0
        assert layer.transform.cache_info().hits == 0

        layer.transform(Xs)
        assert layer.transform.cache_info().misses == 1
        assert layer.transform.cache_info().hits == 0

        Xs2 = tf.placeholder(dtype=tf.float32, shape=(None, 10))
        layer.transform(Xs2)
        assert layer.transform.cache_info().misses == 2
        assert layer.transform.cache_info().hits == 0


class TestLayerAddParam:
    def get_value(self, param):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        config = tf.ConfigProto(gpu_options=gpu_options)
        init = tf.initialize_all_variables()
        with tf.Session(config=config) as session:
            session.run(init)
            session.run(param)
            return param.eval(session)

    @pytest.fixture
    def layer_cls(self):
        from mink.layers import Layer

        class MyLayer(Layer):
            def __init__(self):
                pass
        return MyLayer

    @pytest.fixture
    def layer(self, layer_cls):
        return layer_cls()

    def test_add_param_incorrect_name(self, layer):
        with pytest.raises(ValueError) as exc:
            layer.add_param(
                spec=np.zeros,
                shape=(3, 4),
                name='my_param',
            )

        assert str(exc.value) == (
            "Parameter names should end in '_', e.g. 'W_'.")

    def test_add_param_sets_params(self, layer):
        layer.add_param(
            spec=np.ones,
            shape=(2, 3, 4),
            name='myparam_',
        )
        assert layer.params_['myparam_'] is layer.myparam_

    def test_add_param_no_shape(self, layer):
        with pytest.raises(TypeError) as exc:
            layer.add_param(
                spec=np.ones,
                name='myparam_',
            )
        assert str(exc.value) == 'Cannot add this parameter without a shape.'

    def test_add_param_with_numpy_array_correct_value(self, layer):
        layer.add_param(
            spec=np.ones,
            shape=(2, 3, 4),
            name='myparam_',
        )
        value = self.get_value(layer.myparam_)
        assert np.allclose(value, np.ones((2, 3, 4)))

    def test_add_param_override_no_force(self, layer):
        layer.add_param(
            spec=np.ones,
            shape=(2, 3, 4),
            name='myparam_',
        )
        layer.add_param(
            spec=np.ones,
            shape=(5, 6),
            name='myparam_',
        )
        value = self.get_value(layer.myparam_)
        assert np.allclose(value, np.ones((2, 3, 4)))

    def test_add_param_override_with_force(self, layer):
        layer.add_param(
            spec=np.ones,
            shape=(2, 3, 4),
            name='myparam_',
        )
        layer.add_param(
            spec=np.ones,
            shape=(5, 6),
            name='myparam_',
            force=True,
        )
        value = self.get_value(layer.myparam_)
        assert np.allclose(value, np.ones((5, 6)))

    def test_add_param_with_initializer_correct_value(self, layer):
        from mink.inits import Constant

        layer.add_param(
            spec=Constant(),
            shape=(2, 3, 4),
            name='myparam_',
        )
        value = self.get_value(layer.myparam_)
        assert np.allclose(value, np.zeros((2, 3, 4)))

    def test_add_param_tf_variable_correct_value(self, layer):
        layer.add_param(
            spec=tf.Variable(tf.ones((4, 1))),
            name='myparam_',
        )
        value = self.get_value(layer.myparam_)
        assert np.allclose(value, np.ones((4, 1)))

    def test_add_param_tf_variable_incorrect_shape(self, layer):
        with pytest.raises(ValueError) as exc:
            layer.add_param(
                spec=tf.Variable(tf.ones((4, 1))),
                shape=(1, 4),
                name='myparam_',
            )
        assert str(exc.value) == (
            'Inconsistent shapes: (4, 1) and (1, 4).')

    @pytest.mark.xfail
    def test_variables_share_weight(self, clf_data):
        # Note: Variable sharing does not work yet. Not sure if
        # possible without using tf scopes. As is, the variables are
        # not updated at all, hence the last test fails.
        X, y = clf_data
        arr = np.random.random((20, 20)).astype(np.float32)
        W0 = tf.Variable(arr)

        l0 = layers.InputLayer()
        l1 = layers.DenseLayer(l0, W=W0, num_units=20)
        l2 = layers.DenseLayer(l1, W=W0, num_units=20)
        l3 = layers.DenseLayer(l2)

        from mink import NeuralNetClassifier
        net = NeuralNetClassifier(l3)
        net.fit(X, y, epochs=10)

        w1 = self.get_value(l1.W_)
        w2 = self.get_value(l2.W_)

        assert np.allclose(w1, w2)
        assert not np.allclose(w1, arr)
