"""Load legacy service exports only when a caller requests them.

The standalone memory distribution also uses this package directory, but does
not depend on the agent service, embedding service, or skill catalogue.
"""
from importlib import import_module

__all__ = [
    "SkillMeta",
    "SkillFrontmatter",
    "Skill",
    "SkillUsageRecord",
    "Database",
    "SkillStorage",
    "EmbeddingService",
]


def __getattr__(name):
    modules = {
        "SkillMeta": ".models",
        "SkillFrontmatter": ".models",
        "Skill": ".models",
        "SkillUsageRecord": ".models",
        "Database": ".database",
        "SkillStorage": ".storage",
        "EmbeddingService": ".embeddings",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
