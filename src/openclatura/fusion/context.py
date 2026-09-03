"""Request-scoped policy for systematic fusion planning."""

from __future__ import annotations

from contextvars import ContextVar, Token

from .model import FusionMode

_FUSION_MODE: ContextVar[FusionMode] = ContextVar("openclatura_fusion_mode", default=FusionMode.LEGACY)


def current_fusion_mode() -> FusionMode:
    """Return the fusion policy for the current naming request."""

    return _FUSION_MODE.get()


def set_fusion_mode(mode: FusionMode | str) -> Token[FusionMode]:
    """Install a request-local fusion policy and return its reset token."""

    return _FUSION_MODE.set(FusionMode(mode))


def reset_fusion_mode(token: Token[FusionMode]) -> None:
    """Restore the fusion policy that preceded ``token``."""

    _FUSION_MODE.reset(token)
