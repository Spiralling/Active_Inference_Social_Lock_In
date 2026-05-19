"""Multi-agent paradigm-persistence model on a learned trust network (IWAI26 v2)."""

from src.config import (
    InferenceConfig,
    ModelConfig,
    NetworkConfig,
    PolicyConfig,
    ResourceConfig,
    TrustConfig,
    UtilityConfig,
    WorldConfig,
    build,
)
from src.history import History
from src.population import Population
from src.viz import (
    animate_population,
    cluster_layout,
    detect_communities,
    draw_population,
    fixed_layout,
    society_layout,
    static_strip,
)

__all__ = [
    "animate_population",
    "build",
    "cluster_layout",
    "detect_communities",
    "draw_population",
    "fixed_layout",
    "History",
    "InferenceConfig",
    "ModelConfig",
    "NetworkConfig",
    "PolicyConfig",
    "Population",
    "ResourceConfig",
    "society_layout",
    "static_strip",
    "TrustConfig",
    "UtilityConfig",
    "WorldConfig",
]
