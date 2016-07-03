# pylint: disable=invalid-name,missing-docstring,no-self-use
# pylint: disable=old-style-class,no-init

import pytest


slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)


class TestSklearnCompatibility:
    from mink import NeuralNetClassifier
    from mink import NeuralNetRegressor

    estimators = [
        NeuralNetClassifier,
        NeuralNetRegressor,
    ]

    @pytest.fixture
    def param_grid(self):
        return {
            'update__learning_rate': [0.1, 0.5],
            'max_epochs': [3, 5],
            'dense__num_units': [10, 20],
        }

    @pytest.mark.skip
    @pytest.mark.parametrize('est', estimators)
    def test_sklearn_compatibility(self, est):
        # requires nose
        from sklearn.utils.estimator_checks import check_estimator
        check_estimator(est)

    @slow
    def test_grid_search(self, clf_net, clf_data, param_grid):
        from sklearn.grid_search import GridSearchCV
        X, y = clf_data

        gs = GridSearchCV(
            clf_net,
            param_grid,
            cv=3,
            scoring='accuracy',
            refit=False,
        )
        gs.fit(X, y)


class TestSetParams:
    @pytest.fixture
    def dense_layer_cls(self):
        from mink.layers import DenseLayer
        return DenseLayer

    @pytest.fixture
    def input_layer_cls(self):
        from mink.layers import InputLayer
        return InputLayer

    def test_set_params_2_layers_no_names(
            self, dense_layer_cls, input_layer_cls):
        l0 = input_layer_cls()
        l1 = dense_layer_cls(l0)

        l1.set_params(num_units=567)
        assert l1.num_units == 567

        l1.set_params(incoming__Xs=123)
        assert l0.Xs == 123

    def test_set_params_2_named_layers(self, dense_layer_cls, input_layer_cls):
        l0 = input_layer_cls(name='l0')
        l1 = dense_layer_cls(l0, name='l1')

        l1.set_params(num_units=567)
        assert l1.num_units == 567

        l1.set_params(incoming__Xs=123)
        assert l0.Xs == 123

        l1.set_params(l0__Xs=234)
        assert l0.Xs == 234

    def test_set_params_3_layers_only_first_named(
            self, dense_layer_cls, input_layer_cls):
        l0 = input_layer_cls(name='l0')
        l1 = dense_layer_cls(l0)
        l2 = dense_layer_cls(l1, name='l1')

        l2.set_params(incoming__incoming__Xs=123)
        assert l0.Xs == 123

        l2.set_params(l0__Xs=345)
        assert l0.Xs == 345

    def test_set_params_neural_net_layers_not_named(
            self, dense_layer_cls, input_layer_cls):
        from mink import NeuralNetClassifier

        l0 = input_layer_cls()
        l1 = dense_layer_cls(l0)
        l2 = dense_layer_cls(l1)
        net = NeuralNetClassifier(layer=l2)

        net.set_params(layer__num_units=123)
        assert l2.num_units == 123

        net.set_params(layer__incoming__incoming__Xs=234)
        assert l0.Xs == 234

    def test_set_params_neural_net_named_layers(self, clf_net):
        clf_net.set_params(output__num_units=234)
        assert clf_net.layer.num_units == 234

        clf_net.set_params(dense__num_units=555)
        assert clf_net.layer.incoming.num_units == 555

        clf_net.set_params(input__Xs=432)
        assert clf_net.layer.incoming.incoming.Xs == 432

    @pytest.mark.xfail
    def test_set_params_mixed_named_and_unnamed_layers(
            self, dense_layer_cls, input_layer_cls):
        # The (perhaps irrelevant) use case of mixing named and
        # unnamed layer names in set params does not work at the
        # moment.
        l0 = input_layer_cls(name='l0')
        l1 = dense_layer_cls(l0, name='l1')
        l2 = dense_layer_cls(l1, name='l2')

        l2.set_params(l2__incoming__Xs=777)
        assert l0.Xs == 777
