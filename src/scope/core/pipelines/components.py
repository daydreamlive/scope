# A custom class to give access to components and config in a modular diffusers pipeline blocks
class ComponentsManager:
    def __init__(self, config: dict):
        self.config = DictAccessor(config)

        self._components = {}

    def add(self, name: str, component):
        self._components[name] = component

    def __getattr__(self, name):
        try:
            return self._components[name]
        except KeyError as e:
            raise AttributeError(
                f"'ComponentsManager' object has no attribute '{name}'"
            ) from e


class DictAccessor:
    """A helper to allow attribute-style access to dictionaries."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        try:
            value = self._data[name]
            # recursively wrap nested dicts so you can do config.training.lr etc.
            if isinstance(value, dict):
                return DictAccessor(value)
            return value
        except KeyError as e:
            raise AttributeError(f"No such attribute: {name}") from e

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        """Get value with optional default like dict.get()."""
        return self._data.get(key, default)

    def __repr__(self):
        return repr(self._data)
