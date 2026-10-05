"""Publish compatibility aliases safely for concurrent ``from`` imports."""
import sys
from importlib import import_module


def alias_module(name, namespace, implementation):
    module = import_module(implementation)
    # A waiting ``from alias import symbol`` may retain the original facade
    # object even after sys.modules points to the implementation. Populate
    # both objects before releasing the facade's import lock.
    metadata = {"__name__", "__loader__", "__package__", "__spec__", "__file__",
                "__cached__", "__builtins__"}
    namespace.update({key: value for key, value in vars(module).items() if key not in metadata})
    sys.modules[name] = module
