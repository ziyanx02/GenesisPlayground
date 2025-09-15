def export(cls):
    mod = cls.__module__
    module = __import__(mod)
    if not hasattr(module, "__all__"):
        module.__all__ = []
    module.__all__.append(cls.__name__)
    return cls