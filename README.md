# Mink

A neural network library based on tensorflow and with true sklearn compatibility.

## Goal

In short, the goal is: Lasagne + nolearn - theano + tensorflow = mink

## State

This is in very early stage of development. Help with development is welcome.

## Example

For more elaborate use cases, see notebooks 
[here](https://github.com/BenjaminBossan/mink/blob/master/simple_CNN_01.ipynb)
and
[here](https://github.com/BenjaminBossan/mink/blob/master/simple_example_with_GS.ipynb).

```
from sklearn.datasets import make_classification
from mink import NeuralNetClassifier
from mink import layers

# Get classification data
X, y = make_classification(n_samples=2000, n_classes=5, n_informative=10)

# Define network architecture: no need to set shape of incoming data, 
# number of classes, softmax nonlinearity or anything.

l = layers.InputLayer()
l = layers.DenseLayer(l, name='hidden', num_units=50)
l = layers.DenseLayer(l)

net = NeuralNetClassifier(layer=l)

# It is possible to set the hyperparameters such as the learning 
# rate after initializing the net. If a layer has a name ("hidden"), 
# that name can be used to reference the layer. This allows to easily 
# use GridSearchCV etc.

net.set_params(hidden__num_units=100)
net.set_params(update__learning_rate=0.05)

# Fit the net
net.fit(X, y)

# Make predictions
y_pred = net.predict(X)

```

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
