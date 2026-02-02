"""Minimal plugin module (never actually loaded due to build failure)."""

from scope.core.plugins import hookimpl


@hookimpl
def register_pipelines(register):
    pass  # Never reached - build fails before this can load
