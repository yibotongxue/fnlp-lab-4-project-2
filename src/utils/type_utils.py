def enable_bracket_access(cls):
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{type(self).__name__}' 没有属性 '{key}'")

    def __setitem__(self, key, value):
        setattr(self, key, value)

    cls.__getitem__ = __getitem__
    cls.__setitem__ = __setitem__

    return cls
