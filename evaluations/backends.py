"""Naming backends used by the STOUT-vs-OpenClatura evaluation.

Two backends are exposed, both of which map a SMILES string to an IUPAC name:

* ``openclatura`` - our deterministic namer, installed from PyPI
  (``pip install openclatura``); ``from openclatura import name_smiles``.
* ``stout`` - the neural SMILES-to-IUPAC translator installed from PyPI
  (``pip install STOUT-pypi``); ``from STOUT import translate_forward``.

Both packages live in the same conda environment (``stout-pypi-eval``) so a
single Python process can drive both. The vendored / locally-modified STOUT
lives in a *different* environment and is only used by ``stout_parity.py`` for
the modified-vs-pypi output comparison.

Each loader returns a ``name(smiles: str) -> str`` callable that never raises:
on any failure it returns the empty string, matching how the existing eval
scripts recorded un-nameable molecules.
"""

from __future__ import annotations

from collections.abc import Callable

BACKENDS = ("openclatura", "stout")

# Default JSONL key each backend writes its name under. ``openblue_iupac`` is
# kept for the openclatura backend so the output is drop-in compatible with the
# legacy OpenBlue result files; override with --name-key when desired.
DEFAULT_NAME_KEY = {
    "openclatura": "openblue_iupac",
    "stout": "stout_iupac",
}


def load_openclatura() -> Callable[[str], str]:
    """Return a safe wrapper around ``openclatura.name_smiles``."""
    from openclatura import name_smiles

    def _name(smiles: str) -> str:
        if not smiles:
            return ""
        try:
            return name_smiles(smiles) or ""
        except Exception:
            return ""

    return _name


def load_stout() -> Callable[[str], str]:
    """Return a safe wrapper around the PyPI ``STOUT.translate_forward``."""
    from STOUT import translate_forward

    def _name(smiles: str) -> str:
        if not smiles:
            return ""
        try:
            return translate_forward(smiles) or ""
        except Exception:
            return ""

    return _name


def load_backend(name: str) -> Callable[[str], str]:
    if name == "openclatura":
        return load_openclatura()
    if name == "stout":
        return load_stout()
    raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
