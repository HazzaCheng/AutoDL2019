_global_dict = {}


def reset():
    global _global_dict
    _global_dict = {}


def get(key=None, default=None):
    global _global_dict
    if key is None:
        return _global_dict
    if key not in _global_dict:
        _global_dict[key] = default
    return _global_dict[key]


def put(key, value):
    global _global_dict
    if isinstance(key, list):
        cur_dict = _global_dict
        keys = key
        while len(keys) > 1:
            cur_dict = cur_dict[keys[0]]
            keys = keys[1:]
        cur_dict[keys[0]] = value
    else:
        _global_dict[key] = value
