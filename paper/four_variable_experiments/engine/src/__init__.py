"""Minimal production exports for the structural branch."""

from importlib import import_module

from src.config import NetworkConfig
from src.network import build_adjacency

__all__ = ["NetworkConfig", "build_adjacency", "structural"]


def __getattr__(name: str):
    if name == "structural":
        return import_module("src.structural")
    raise AttributeError(f"module 'src' has no attribute {name!r}")
