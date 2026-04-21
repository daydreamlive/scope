"""Eval harness for the Scope agentic workflow builder.

Each "case" is a YAML file in ``evals/cases/`` describing a natural-language
prompt, how many times to sample the model, and structural checks to run on
the resulting workflow proposal. The runner drives the real agent via an
in-process ASGI transport and grades proposals deterministically.

This package is NOT imported by the running server; it is only exercised by
``python -m evals`` (CLI) and the opt-in ``pytest -m eval`` smoke test.
"""
