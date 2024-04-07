def update_kwargs(kwargs, defaults):
    """
    Update the given keyword arguments with default values.

    This function takes a dictionary of keyword arguments (`kwargs`) and a dictionary of default values (`defaults`).
    It iterates over the keys in the `defaults` dictionary and checks if each key is present in the `kwargs` dictionary.
    If a key is not present in `kwargs`, it adds the key-value pair from `defaults` to `kwargs`.

    Args:
        kwargs (dict): The keyword arguments to be updated.
        defaults (dict): The dictionary of default values.

    Returns:
        dict: The updated keyword arguments.

    Example:
        >>> kwargs = {'a': 1, 'b': 2}
        >>> defaults = {'b': 3, 'c': 4}
        >>> update_kwargs(kwargs, defaults)
        {'a': 1, 'b': 2, 'c': 4}
    """
    for key in defaults:
        if key not in kwargs:
            kwargs[key] = defaults[key]
    return kwargs
