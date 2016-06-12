import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import tosequence


class IteratorPipeline(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            batch_size,
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

    def fit(self, X, y, **kwargs):
        transform_steps = dict((step, {}) for step, __ in self.steps)
        for pname, pval in kwargs.items():
            step, param = pname.split('__', 1)
            transform_steps[step][param] = pval

        for name, transform in self.steps:
            transform.fit(X, y, **transform_steps[name])
        return self

    def transform(self, X, y=None, **kwargs):
        Xt, yt = X, y
        for __, transform in self.steps:
            Xt, yt = transform.transform(
                Xt,
                yt,
                deterministic=self.deterministic,
                **kwargs)
        return Xt, yt

    def iter_transform(self, X, y=None, **kwargs):
        bs = self.batch_size
        Xt = []
        if y is not None:
            yt = []
        else:
            yt = y

        for i in range((X.shape[0] + bs - 1) // bs):
            Xb = X[i * bs:(i + 1) * bs]
            if y is not None:
                yb = y[i * bs:(i + 1) * bs]
            else:
                yb = None

            Xbt, ybt = self.transform(Xb, yb, **kwargs)
            Xt.append(Xbt)
            yt.append(ybt)
        return np.concatenate(Xt), np.concatenate(yt)

    def fit_transform(self, X, y, **kwargs):
        return self.fit(
            X, y, **kwargs
        ).transform(
            X, y, **kwargs)

    def __call__(self, X, y=None):
        self.X_ = X
        self.y_ = y
        return self

    def __iter__(self):
        X, y, bs = self.X_, self.y_, self.batch_size
        for i in range((X.shape[0] + bs - 1) // bs):
            Xb = X[i * bs:(i + 1) * bs]
            if y is not None:
                yb = y[i * bs:(i + 1) * bs]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in state:
            if attr.endswith('_'):
                del state[attr]
        return state


class Iterator(BaseEstimator, TransformerMixin):
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
