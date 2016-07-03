"""Contains batch iterators and the iterator pipeline to chain
iterators.

"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import tosequence


class IteratorPipeline(BaseEstimator, TransformerMixin):
    """A pipeline that is responsible for batching and, optionally,
    applying transforms on a batch level. Similar in use to an
    sklearn.pipeline.Pipeline.

    Should be used with the iterator protocol.

    """
    def __init__(
            self,
            batch_size=128,
            steps=[],
            deterministic=False,
    ):
        self.batch_size = batch_size
        self.steps = steps
        self.deterministic = deterministic

        if steps:
            self._check_steps(steps)

    def _check_steps(self, steps):
        names, transforms = zip(*steps)
        if len(dict(steps)) != len(steps):
            raise ValueError("Provided step names are not unique: {}"
                             "".format(names))

        # shallow copy of steps
        self.steps = tosequence(steps)

        for t in transforms:
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All steps of the chain should "
                                "be transforms and implement fit and transform"
                                " '%s' (type %s) doesn't)" % (t, type(t)))

    def get_params(self, deep=True):
        """TODO"""
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_steps
            for name, step in self.named_steps.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value

            out.update(super().get_params(deep=False))
            return out

    @property
    def named_steps(self):
        return dict(self.steps)

    def fit(self, X, y, **kwargs):
        """TODO"""
        transform_steps = dict((step, {}) for step, __ in self.steps)
        for pname, pval in kwargs.items():
            step, param = pname.split('__', 1)
            transform_steps[step][param] = pval

        for name, transform in self.steps:
            transform.fit(X, y, **transform_steps[name])
        return self

    def transform(self, X, y=None, **kwargs):
        """TODO"""
        Xt, yt = X, y
        for _, transform in self.steps:
            Xt, yt = transform.transform(
                Xt,
                yt,
                deterministic=self.deterministic,
                **kwargs)
        return Xt, yt

    def iter_transform(self, X, y=None, **kwargs):
        """Batch all X (and y) and apply transforms.

        This could be useful if you want to inspect the transformed
        data.

        """
        batch_size = self.batch_size
        Xt = []
        if y is not None:
            yt = []
        else:
            yt = y

        for i in range((X.shape[0] + batch_size - 1) // batch_size):
            Xb = X[i * batch_size:(i + 1) * batch_size]
            if y is not None:
                yb = y[i * batch_size:(i + 1) * batch_size]
            else:
                yb = None

            Xbt, ybt = self.transform(Xb, yb, **kwargs)
            Xt.append(Xbt)
            yt.append(ybt)
        return np.concatenate(Xt), np.concatenate(yt)

    def fit_transform(self, X, y, **kwargs):
        """TODO"""
        return self.fit(
            X, y, **kwargs
        ).transform(
            X, y, **kwargs)

    def __call__(self, X, y=None):
        # pylint: disable=attribute-defined-outside-init
        self.X_ = X
        self.y_ = y
        return self

    def __iter__(self):
        X, y, batch_size = self.X_, self.y_, self.batch_size
        for i in range((X.shape[0] + batch_size - 1) // batch_size):
            Xb = X[i * batch_size:(i + 1) * batch_size]
            if y is not None:
                yb = y[i * batch_size:(i + 1) * batch_size]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in state:
            if attr.endswith('_'):
                del state[attr]
        return state


def _identity(X):
    return X


class Iterator(BaseEstimator, TransformerMixin):
    """Iterator base class."""
    def fit(self, X, y, **kwargs):
        return self

    def fit_transform(self, X, y, deterministic, **kwargs):
        return self.fit(
            X, y, **kwargs
        ).transform(
            X, y, deterministic=deterministic, **kwargs)

    def transform(self, X, y, deterministic, **kwargs):
        raise NotImplementedError


class GaussianNoiseIterator(Iterator):
    """Apply gaussian noise to data."""
    def __init__(
            self,
            mean=0.0,
            std=1.0,
    ):
        self.mean = mean
        self.std = std

    def transform(self, X, y, deterministic, **kwargs):
        if not deterministic:
            return X, y
        else:
            noise = np.random.randn(*X.shape)
            noise *= self.std
            noise += self.mean
            return X + noise.astype(X.dtype), y


class FunctionIterator(Iterator):
    def __init__(
            self,
            func=None,
            func_deterministic=None,
    ):
        self.func = func
        self.func_deterministic = func_deterministic

    def transform(self, X, y, deterministic, **kwargs):
        func = self.func or _identity
        func_deterministic = self.func_deterministic or func

        if deterministic:
            return func_deterministic(X), y
        else:
            return func(X), y
