"""Resolve the package version from the code that is actually being imported."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

import tomllib


def _source_tree_version() -> str | None:
    """Return this checkout's project version when imported from ``src``."""

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle).get("project")
    except (OSError, tomllib.TOMLDecodeError):
        return None

    if not isinstance(project, dict) or project.get("name") != "openclatura":
        return None
    version = project.get("version")
    return version if isinstance(version, str) else None


def resolve_version() -> str:
    """Prefer source metadata, then metadata for an installed distribution."""

    source_version = _source_tree_version()
    if source_version is not None:
        return source_version
    try:
        return distribution_version("openclatura")
    except PackageNotFoundError:
        return "unknown"


__version__ = resolve_version()
