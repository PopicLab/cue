from collections import defaultdict


class NestedDict(defaultdict):
    def __call__(self):
        return NestedDict(self.default_factory)
