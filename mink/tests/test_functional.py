import pickle

import numpy as np
import pytest
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from unittest.mock import patch

from mink import NeuralNetClassifier
from mink.layers import DenseLayer
from mink.layers import InputLayer


class TestSaveLoadModel:
    def test_pickle_save_load(
            self, clf_net, clf_data, _layers, session_kwargs, tmpdir):
        X, y = clf_data
        clf_net.fit(X, y, num_epochs=10)
        score_before = accuracy_score(y, clf_net.predict(X))

        p = tmpdir.mkdir('mink').join('testmodel.ckpt')
        with open(str(p), 'wb') as f:
            pickle.dump(clf_net, f)
        del clf_net

        with open(str(p), 'rb') as f:
            new_net = pickle.load(f)

        new_net.initialize(X, y)
        score_after = accuracy_score(y, new_net.predict(X))
        assert np.isclose(score_after, score_before)

    def test_load_params_from_other_model(
            self, clf_net, clf_data, _layers, session_kwargs):
        X, y = clf_data
        clf_net.fit(X, y, num_epochs=10)
        score_before = accuracy_score(y, clf_net.predict(X))

        new_net = NeuralNetClassifier(_layers)
        new_net.initialize(X, y)
        score_init = accuracy_score(y, new_net.predict(X))
        assert not np.isclose(score_init, score_before)

        new_net.set_all_params(clf_net.get_all_params())
        score_after = accuracy_score(y, new_net.predict(X))
        assert np.isclose(score_after, score_before)


def test_call_fit_with_custom_session_kwargs(_layers, clf_data):
    X, y = clf_data
    session_kwargs = {'a': 'b'}

    with patch('mink.base.tf.Session') as mock_session:
        from mink.base import NeuralNetClassifier
        net = NeuralNetClassifier(_layers, session_kwargs=session_kwargs)
        net.initialize(X, y)

        assert mock_session.call_args_list[0][1] == session_kwargs


def test_call_fit_repeatedly(clf_net, clf_data):
    X, y = clf_data

    clf_net.fit(X, y, num_epochs=15)
    accuracy_before = (y == clf_net.predict(X)).mean()

    clf_net.fit(X, y, num_epochs=5)
    accuracy_after = (y == clf_net.predict(X)).mean()

    # after continuing fit, accuracy should decrease
    assert accuracy_after < accuracy_before


class TestSetParams:
    def test_set_params_2_layers_no_names(self):
        l0 = InputLayer()
        l1 = DenseLayer(l0)

        l1.set_params(num_units=567)
        assert l1.num_units == 567

        l1.set_params(incoming__Xs=123)
        assert l0.Xs == 123

    def test_set_params_2_named_layers(self):
        l0 = InputLayer(name='l0')
        l1 = DenseLayer(l0, name='l1')

        l1.set_params(num_units=567)
        assert l1.num_units == 567

        l1.set_params(incoming__Xs=123)
        assert l0.Xs == 123

        l1.set_params(l0__Xs=234)
        assert l0.Xs == 234

    def test_set_params_3_layers_only_first_named(self):
        l0 = InputLayer(name='l0')
        l1 = DenseLayer(l0)
        l2 = DenseLayer(l1, name='l1')

        l2.set_params(incoming__incoming__Xs=123)
        assert l0.Xs == 123

        l2.set_params(l0__Xs=345)
        assert l0.Xs == 345

    def test_set_params_neural_net_layers_not_named(self):
        l0 = InputLayer()
        l1 = DenseLayer(l0)
        l2 = DenseLayer(l1)
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
    def test_set_params_mixed_named_and_unnamed_layers(self):
        # The (perhaps irrelevant) use case of mixing named and
        # unnamed layer names in set params does not work at the
        # moment.
        l0 = InputLayer(name='l0')
        l1 = DenseLayer(l0, name='l1')
        l2 = DenseLayer(l1, name='l2')

        l2.set_params(l2__incoming__Xs=777)
        assert l0.Xs == 777


class TestNeuralNetClassifier:
    def test_neural_net_classifier_learns(self, clf_net, clf_data):
        X, y = clf_data

        clf_net.fit(X, y, num_epochs=0)
        score_before = accuracy_score(y, clf_net.predict(X))
        assert np.isclose(score_before, 1.0 / len(np.unique(y)), rtol=0.3)

        clf_net.fit(X, y, num_epochs=50)
        score_after = accuracy_score(y, clf_net.predict(X))
        min_improvement = score_before * (1 - score_before)
        assert score_after > score_before + min_improvement


class TestNeuralNetRegressor:
    def test_neural_net_regressor_learns(self, regr_net, regr_data):
        X, y = regr_data

        regr_net.fit(X, y, num_epochs=0)
        score_before = mean_squared_error(y, regr_net.predict(X))
        assert np.isclose(score_before, (y ** 2).mean(), rtol=0.2)

        regr_net.fit(X, y, num_epochs=10)
        score_after = mean_squared_error(y, regr_net.predict(X))
        assert score_after < 0.01 * score_before


class TestNetIteratorPipeline:
    @pytest.fixture
    def iterators(self):
        from mink import iterators
        pipe_train = iterators.IteratorPipeline(
            steps=[('noise0', iterators.GaussianNoiseIterator()),
                   ('noise1', iterators.GaussianNoiseIterator())],
            deterministic=False,
        )
        pipe_test = iterators.IteratorPipeline(
            steps=[('noise0', iterators.GaussianNoiseIterator()),
                   ('noise1', iterators.GaussianNoiseIterator())],
            deterministic=True,
        )
        return pipe_train, pipe_test

    def test_net_with_iterators_as_ints(self, clf_net, clf_data):
        X, y = clf_data
        clf_net.batch_iterator_train = 74
        clf_net.batch_iterator_test = 229

        # does not raise
        clf_net.fit(X, y, num_epochs=3)
        clf_net.predict(X)

    def test_net_with_iterator_pipeline(self, clf_net, clf_data, iterators):
        X, y = clf_data
        clf_net.batch_iterator_train = iterators[0]
        clf_net.batch_iterator_test = iterators[1]

        # does not raise
        clf_net.fit(X, y, num_epochs=3)
        clf_net.predict(X)

    def test_net_with_iterator_pipeline_set_params(
            self, clf_net, clf_data, iterators):
        X, y = clf_data
        clf_net.batch_iterator_train = iterators[0]
        clf_net.batch_iterator_test = iterators[1]

        clf_net.set_params(batch_iterator_train__noise0__mean=10)
        clf_net.set_params(batch_iterator_test__noise1__std=-2)

        # does not raise
        clf_net.fit(X, y, num_epochs=3)
        clf_net.predict(X)


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

    def test_grid_search(self, clf_net, clf_data, param_grid):
        X, y = clf_data

        gs = GridSearchCV(
            clf_net,
            param_grid,
            cv=3,
            scoring='accuracy',
            refit=False,
        )
        gs.fit(X, y)
