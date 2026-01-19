"""Shared utilities for patch modules."""

import importlib.util
import os


def find_package_path(package_name: str) -> str | None:
    """Find a package's install path WITHOUT importing it.

    This is critical for torch - importing it loads DLLs which then
    can't be overwritten. Using find_spec() locates the package without
    executing its __init__.py.

    Handles both regular packages (with __init__.py) and namespace packages.
    """
    try:
        spec = importlib.util.find_spec(package_name)
        if spec:
            # Regular package: spec.origin points to __init__.py
            if spec.origin:
                return os.path.dirname(spec.origin)
            # Namespace package: use submodule_search_locations
            if spec.submodule_search_locations:
                locations = list(spec.submodule_search_locations)
                if locations:
                    return locations[0]
    except (ImportError, ModuleNotFoundError):
        pass
    return None
