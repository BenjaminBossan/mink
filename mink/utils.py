import numpy as np
import tensorflow as tf


def as_tuple(x, N, t=None):
    """Coerce a value to a tuple of given length (and possibly given
    type).

    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type, optional
        required type for all elements

    Returns
    -------
    tuple
        ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.

    Raises
    ------
    TypeError
        if `type` is given and `x` or any of its elements do not match it

    ValueError
        if `x` is iterable, but does not have exactly `N` elements

    """
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X


def as_4d(x):
    if not isinstance(x, (list, tuple)):
        return (1, x, x, 1)
    else:
        return x


def get_shape(placeholder):
    return tuple(placeholder.get_shape().as_list())


def set_named_layer_param(layer, key, value):
    name = layer.name
    start, end = key.split('__', 1)

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
    shape = list(get_shape(Xs))
    last_dim = np.prod(shape[ndim - 1:])
    if shape[0] is None:
        new_shape = [-1] + shape[1:ndim - 1] + [last_dim]
    else:
        new_shape = shape[:ndim - 1] + [last_dim]
    return tf.reshape(Xs, new_shape)


def get_incomings(layer):
    incoming = getattr(layer, 'incoming', None)
    if incoming:
        return [incoming]
    incomings = getattr(layer, 'incomings', [])
    return incomings


def get_all_layers(layer):
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
    return known


def get_input_layers(layer):
    input_layers = []
    all_layers = get_all_layers(layer)
    for layer in all_layers:
        if (
                (not hasattr(layer, 'incoming')) and
                (not hasattr(layer, 'incomings'))
        ):
            input_layers.append(layer)
    return input_layers
