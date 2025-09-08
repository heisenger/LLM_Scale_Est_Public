def get_explicit_class_vars(cls):
    return {
        k: v
        for k, v in cls.__dict__.items()
        if not (k.startswith("__") and k.endswith("__")) and not callable(v)
    }
