# pylint: disable=invalid-name,missing-docstring,no-self-use
# pylint: disable=old-style-class,no-init

import pytest

from mink import layers


class TestGetLayers:
    @pytest.fixture
    def get_all_layers(self):
        from mink.utils import get_all_layers
        return get_all_layers

    def test_get_all_layers_linear(self, get_all_layers):
        C = layers.InputLayer()
        B = layers.DenseLayer(C)
        A = layers.DenseLayer(B)

        assert get_all_layers(C) == [C]
        assert get_all_layers(B) == [C, B]
        assert get_all_layers(A) == [C, B, A]

    def test_get_all_layers_with_concat(self, get_all_layers):
        B = layers.InputLayer()
        E = layers.InputLayer()
        D = layers.InputLayer()
        C = layers.DenseLayer(E)
        A = layers.ConcatLayer([B, C, D])

        assert get_all_layers(A) == [A, B, C, D, E][::-1]

    def test_get_all_layers_several_concats(self, get_all_layers):
        J = layers.InputLayer()
        K = layers.InputLayer()
        G = layers.InputLayer()
        H = layers.ConcatLayer([J, K])
        I = layers.DenseLayer(K)
        D = layers.ConcatLayer([G, H, I])
        E = layers.DenseLayer(I)
        F = layers.InputLayer()
        B = layers.DenseLayer(D)
        C = layers.ConcatLayer([E, F])
        A = layers.ConcatLayer([B, C])

        assert get_all_layers(D) == [D, G, H, I, J, K][::-1]
        assert get_all_layers(A) == [A, B, C, D, E, F, G, H, I, J, K][::-1]

    def test_get_all_layers_with_list(self, get_all_layers):
        B = layers.InputLayer()
        E = layers.InputLayer()
        D = layers.InputLayer()
        C = layers.DenseLayer(E)
        A = layers.ConcatLayer([B, C, D])

        lst = [D, C, E, A, B]
        assert get_all_layers(lst) == lst

        lst = [A, B, C, D, E]
        assert get_all_layers(lst) == lst

        lst = [B, E, D, C, A]
        assert get_all_layers(lst) == lst

    @pytest.fixture
    def get_input_layers(self):
        from mink.utils import get_input_layers
        return get_input_layers

    def test_get_input_layers(self, get_input_layers):
        C = layers.InputLayer()
        B = layers.DenseLayer(C)
        A = layers.DenseLayer(B)

        assert get_input_layers(C) == [C]
        assert get_input_layers(B) == [C]
        assert get_input_layers(A) == [C]

    def test_get_input_layers_with_concat(self, get_input_layers):
        B = layers.InputLayer()
        E = layers.InputLayer()
        D = layers.InputLayer()
        C = layers.DenseLayer(E)
        A = layers.ConcatLayer([B, C, D])

        assert get_input_layers(A) == [E, D, B]

    def test_get_input_layers_several_concats(self, get_input_layers):
        J = layers.InputLayer()
        K = layers.InputLayer()
        G = layers.InputLayer()
        H = layers.ConcatLayer([J, K])
        I = layers.DenseLayer(K)
        D = layers.ConcatLayer([G, H, I])
        E = layers.DenseLayer(I)
        F = layers.InputLayer()
        B = layers.DenseLayer(D)
        C = layers.ConcatLayer([E, F])
        A = layers.ConcatLayer([B, C])

        assert get_input_layers(D) == [G, J, K][::-1]
        assert get_input_layers(A) == [F, G, J, K][::-1]
