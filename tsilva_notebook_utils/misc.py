def filter_kwargs(func, kwargs):
    import inspect
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}
