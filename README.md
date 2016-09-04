# Mink

A neural network library based on tensorflow and with true sklearn compatibility.

## Goal

In short, the goal is: Lasagne + nolearn - theano + tensorflow = mink

## State

This is in very early stage of development. Help with development is welcome.

## Example

For more elaborate use cases, see notebooks 
[here (CNN)](https://github.com/BenjaminBossan/mink/blob/master/notebooks/simple_CNN_01.ipynb),
[here (RNN)](https://github.com/BenjaminBossan/mink/blob/master/notebooks/simple_RNN.ipynb),
[here (grid search)](https://github.com/BenjaminBossan/mink/blob/master/notebooks/simple_example_with_GS.ipynb), and
[here (validation scores)](https://github.com/BenjaminBossan/mink/blob/master/notebooks/using_validation_sets.ipynb).

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
