"""Multi-agent active-inference coupling layer for variational paradigm dynamics."""

from src.config import ModelConfig, build
from src.environment import CoordinationEnvironment, EnvConfig, EvolutionRegime
from src.history import History
from src.population import Population
from src.viz import (
    animate_population,
    draw_population,
    fixed_layout,
    static_strip,
)

__all__ = [
    "animate_population",
    "build",
    "CoordinationEnvironment",
    "draw_population",
    "EnvConfig",
    "EvolutionRegime",
    "fixed_layout",
    "History",
    "ModelConfig",
    "Population",
    "static_strip",
]
