# pylint: disable=invalid-name,missing-docstring,no-self-use
# pylint: disable=old-style-class,no-init

import pickle
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


class TestSaveLoadModel:
    @pytest.fixture
    def net_cls(self):
        from mink import NeuralNetClassifier
        return NeuralNetClassifier

    def test_pickle_save_load(self, clf_net, clf_data, tmpdir):
        X, y = clf_data
        clf_net.fit(X, y, num_epochs=10)
        score_before = accuracy_score(y, clf_net.predict(X))

        p = tmpdir.mkdir('mink').join('testmodel.ckpt')
        with open(str(p), 'wb') as f:
            pickle.dump(clf_net, f)
        del clf_net

        with open(str(p), 'rb') as f:
            new_net = pickle.load(f)

        score_after = accuracy_score(y, new_net.predict(X))
        assert np.isclose(score_after, score_before)

    def test_load_params_from_other_model(
            self, net_cls, clf_net, clf_data, _layers):
        X, y = clf_data
        clf_net.fit(X, y, num_epochs=10)
        score_before = accuracy_score(y, clf_net.predict(X))

        new_net = net_cls(_layers)
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


class TestNeuralNetEstimatorsLearn:
    def test_neural_net_classifier_learns(self, clf_net, clf_data):
        X, y = clf_data

        clf_net.fit(X, y, num_epochs=0)
        score_before = accuracy_score(y, clf_net.predict(X))
        assert np.isclose(score_before, 1.0 / len(np.unique(y)), rtol=0.3)

        clf_net.fit(X, y, num_epochs=50)
        score_after = accuracy_score(y, clf_net.predict(X))
        min_improvement = score_before * (1 - score_before)
        assert score_after > score_before + min_improvement

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
