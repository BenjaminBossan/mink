## Design

To achieve the goal of true sklearn compatibility, mink defines a
layer to correspond to an sklearn transformer. This can be seen
[here](https://github.com/BenjaminBossan/mink/blob/master/analogy_to_sklearn.ipynb).

Layers, as well as other components of mink (updates, nonlinearities,
etc.), all inherit from sklearn's BaseEstimator. This allows us to use
them in conjunction with sklearn tools such as GridSearchCV.

Neural networks form directed acyclical graphs (DAGs), whereas sklearn
only supports very linear combinations and estimators and
transformers. Layers implement an `initialize` and a `get_output`
method. These are equivalent to the `fit` and `transform` methods in
sklearn, but instead of *pushing* the data through the pipeline, we
recursively call incoming layers of the final layer with tensorflow
symbolic variables, i.e. we *pull* the output from the incoming
layers. This way, we are free to define arbitrary DAGs.
