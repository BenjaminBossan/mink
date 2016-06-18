"""Contains base utilities that don't depend on other modules."""

import numpy as np
import tensorflow as tf


def as_tuple(value, num, dtype=None):
    """Coerce a value to a tuple of given length (and possibly given
    type).

    Parameters
    ----------
    value : value or iterable
    num : integer
        length of the desired tuple
    dtype : type, optional
        required type for all elements

    Returns
    -------
    tuple
        ``tuple(value)`` if `value` is iterable, ``(value,) * num`` otherwise.

    Raises
    ------
    TypeError
        if `dtype` is given and `value` or any of its elements do not match it

    ValueError
        if `value` is iterable, but does not have exactly `num` elements

    """
    try:
        X = tuple(value)
    except TypeError:
        X = (value,) * num

    if (dtype is not None) and not all(isinstance(v, dtype) for v in X):
        raise TypeError(
            "expected a single value or an iterable "
            "of {0}, got {1} instead".format(dtype.__name__, value))

    if len(X) != num:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(num, value))

    return X


def as_4d(value):
    """Return a tuple of length 4 (1, value, value, 1) if not a tuple
    already.

    """
    if not isinstance(value, (list, tuple)):
        return (1, value, value, 1)
    elif len(value) == 4:
        return value
    else:
        raise ValueError("Cannot transform to length 4 tuple.")


def get_shape(placeholder):
    return tuple(placeholder.get_shape().as_list())


def set_named_layer_param(layer, key, value):
    """TODO"""
    name = layer.name
    end = key.split('__', 1)[1]

    if (
            not name or
            not key.startswith(name) or
            not hasattr(layer, end)
    ):
        # if not possible to set on this layer, try incoming layer
        incoming = getattr(layer, 'incoming', None)
        if not incoming:
            # dead end
            return False
        else:
            # try to set on incoming layer
            return set_named_layer_param(incoming, key, value)

    setattr(layer, end, value)
    return True


def flatten(Xs, ndim=2):
    """Flatten a symbolic variable to given number of dimensions."""
    shape = list(get_shape(Xs))
    last_dim = np.prod(shape[ndim - 1:])
    if shape[0] is None:
        new_shape = [-1] + shape[1:ndim - 1] + [last_dim]
    else:
        new_shape = shape[:ndim - 1] + [last_dim]
    return tf.reshape(Xs, new_shape)


def get_incomings(layer):
    """Get all incoming layers for given layer."""
    incoming = getattr(layer, 'incoming', None)
    if incoming:
        return [incoming]
    incomings = getattr(layer, 'incomings', [])
    return incomings


def get_all_layers(layer):
    """Perform a breadth-first-search for all layers incoming to this
    layer.

    """
    # Assumes that all layers are in list. Maybe check?
    if isinstance(layer, list):
        return layer

    known = []
    incomings = [layer]
    while incomings:
        layer = incomings.pop(0)
        if layer not in known:
            known.append(layer)
        incomings += get_incomings(layer)
    return known[::-1]


def get_input_layers(layer):
    """Find all layers in the graph descending to this layer that are
    InputLayers.

    """
    input_layers = []
    all_layers = get_all_layers(layer)
    for layer in all_layers:
        if (
                (not hasattr(layer, 'incoming')) and
                (not hasattr(layer, 'incomings'))
        ):
            input_layers.append(layer)
    return input_layers


def get_layer_name(layer):
    """Return the name of the layer or create one if there is none."""
    if layer.name:
        return layer.name
    name = layer.__class__.__name__.split('.')[-1]
    if name.endswith('Layer'):
        name = name[:-5]
    return name.lower()
