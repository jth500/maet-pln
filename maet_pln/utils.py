def update_kwargs(kwargs, defaults):
    for key in defaults:
        if key not in kwargs:
            kwargs[key] = defaults[key]
    return kwargs
