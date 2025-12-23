"""Core functionality for Scope."""


def __getattr__(name):
    """Lazy import for plugin hooks to avoid requiring pluggy in worker environments."""
    if name == "hookimpl":
        from scope.core.plugins import hookimpl

        return hookimpl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["hookimpl"]
