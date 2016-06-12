import numpy as np
import pytest
from sklearn.base import clone


class TestIteartorPipeline:
    @pytest.fixture
    def iterator_pipeline_cls(self):
        from mink.iterators import IteratorPipeline
        return IteratorPipeline

    @pytest.fixture(scope='function')
    def add1_iterator(self):
        from mink.iterators import Iterator

        class Add1Iterator(Iterator):
            def transform(self, X, y, deterministic, **kwargs):
                if not hasattr(self, '_count'):
                    self._count = 0

                self._count += 1
                if deterministic:
                    return X, y
                else:
                    return X + 1, y

        return Add1Iterator()

    def get_iterator_pipeline(
            self,
            batch_size=128,
            n_steps=2,
            deterministic=False,
    ):
        iterator_pipeline_cls = self.iterator_pipeline_cls()
        add1_iterator = self.add1_iterator()
        X, y = self.data()

        return iterator_pipeline_cls(
            batch_size=batch_size,
            steps=[(str(i), clone(add1_iterator)) for i in range(n_steps)],
            deterministic=deterministic,
        )

    @pytest.fixture
    def data(self):
        X = np.zeros((1000, 100))
        y = np.zeros(1000)
        return X, y

    @pytest.mark.parametrize('n_steps', [0, 1, 2, 5])
    def test_iterator_pipeline_deterministic(self, data, n_steps):
        X, y = data
        pipe = self.get_iterator_pipeline(
            n_steps=n_steps,
            deterministic=True,
        )
        pipe.fit(X, y)
        Xt, yt = pipe.transform(X, y)

        assert np.allclose(Xt, np.zeros_like(X))
        assert np.allclose(yt, np.zeros_like(y))

    @pytest.mark.parametrize('n_steps', [0, 1, 2, 5])
    def test_iterator_pipeline_deterministic_fit_transform(
            self, data, n_steps):
        X, y = data
        pipe = self.get_iterator_pipeline(
            n_steps=n_steps,
            deterministic=True,
        )
        Xt, yt = pipe.fit_transform(X, y)

        assert np.allclose(Xt, np.zeros_like(X))

    @pytest.mark.parametrize('n_steps', [0, 1, 2, 5])
    def test_iterator_pipeline_deterministic_iter_transform(
            self, data, n_steps):
        X, y = data
        pipe = self.get_iterator_pipeline(
            n_steps=n_steps,
            deterministic=True,
        )
        pipe.fit(X, y)
        Xt, yt = pipe.iter_transform(X, y)

        assert np.allclose(Xt, np.zeros_like(X))

    @pytest.mark.parametrize('n_steps', [0, 1, 2, 5])
    def test_iterator_pipeline_non_deterministic(self, data, n_steps):
        X, y = data
        pipe = self.get_iterator_pipeline(
            n_steps=n_steps,
            deterministic=False,
        )
        pipe.fit(X, y)
        Xt, yt = pipe.transform(X, y)

        assert np.allclose(Xt, np.zeros_like(X) + n_steps)

    @pytest.mark.parametrize('n_steps', [0, 1, 2, 5])
    def test_iterator_pipeline_non_deterministic_fit_transform(
            self, data, n_steps):
        X, y = data
        pipe = self.get_iterator_pipeline(
            n_steps=n_steps,
            deterministic=False,
        )
        Xt, yt = pipe.fit_transform(X, y)

        assert np.allclose(Xt, np.zeros_like(X) + n_steps)

    @pytest.mark.parametrize('n_steps', [0, 1, 2, 5])
    def test_iterator_pipeline_non_deterministic_iter_transform(
            self, data, n_steps):
        X, y = data
        pipe = self.get_iterator_pipeline(
            n_steps=n_steps,
            deterministic=False,
        )
        pipe.fit(X, y)
        Xt, yt = pipe.iter_transform(X, y)

        assert np.allclose(Xt, np.zeros_like(X) + n_steps)

    @pytest.mark.parametrize('batch_size', [1, 50, 256, 1001, 2000])
    def test_iter_number_of_iterations(self, data, batch_size):
        X, y = data
        pipe = self.get_iterator_pipeline(batch_size=batch_size)
        pipe.fit(X, y)
        for __ in pipe(X, y):
            pass

        expected_count = X.shape[0] // batch_size
        if X.shape[0] % batch_size > 0:
            expected_count += 1

        for __, transform in pipe.steps:
            assert transform._count == expected_count

    @pytest.mark.parametrize('batch_size', [1, 50, 256, 1001, 2000])
    def test_transform_number_of_iterations(self, data, batch_size):
        X, y = data
        pipe = self.get_iterator_pipeline(batch_size=batch_size)
        pipe.fit(X, y)
        pipe.transform(X, y)

        for __, transform in pipe.steps:
            assert transform._count == 1

    @pytest.mark.parametrize('batch_size', [1, 50, 256, 1001, 2000])
    def test_fit_transform_number_of_iterations(self, data, batch_size):
        X, y = data
        pipe = self.get_iterator_pipeline(batch_size=batch_size)
        pipe.fit_transform(X, y)

        for __, transform in pipe.steps:
            assert transform._count == 1

    @pytest.mark.parametrize('n_steps', [0, 1, 2, 5])
    @pytest.mark.parametrize('batch_size', [1, 50, 256, 1001, 2000])
    def test_iter_transform_number_of_iterations(
            self, data, n_steps, batch_size):
        X, y = data
        pipe = self.get_iterator_pipeline(
            batch_size=batch_size,
            n_steps=n_steps,
        )
        pipe.fit(X, y)
        Xt, yt = pipe.iter_transform(X, y)

        expected_count = X.shape[0] // batch_size
        if X.shape[0] % batch_size > 0:
            expected_count += 1

        for __, transform in pipe.steps:
            assert transform._count == expected_count
