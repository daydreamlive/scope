"""Minimal plugin module (never actually loaded due to compile failure)."""

from scope.core.plugins import hookimpl


@hookimpl
def register_pipelines(register):
    pass  # Never reached - compile fails before this can load
