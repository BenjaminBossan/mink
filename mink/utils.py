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
            split = key.split('__')
            return set_named_layer_param(incoming, key, value)

    setattr(layer, end, value)
    return True
