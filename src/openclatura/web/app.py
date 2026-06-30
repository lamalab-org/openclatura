"""FastAPI HTTP surface for openclatura.

Endpoints:

- ``GET  /healthz``      → ``{"ok": true, "version": "..."}``
- ``POST /name``         → name one SMILES, returns ``NamingResult.to_dict()``
- ``POST /batch``        → name many SMILES, returns ``list[NamingResult.to_dict()]``
- ``POST /describe``     → render natural-language ``Description.to_dict()``

Run locally::

    uvicorn openclatura.web.app:app --host 0.0.0.0 --port 8000

Or via the helper::

    python -m openclatura.web

Requires the ``[web]`` extra. ``[opsin]`` + a JDK are needed for
``verify_opsin=True``; otherwise the endpoint reports the OPSIN
skipped status.
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - extra not installed
    raise ImportError("openclatura.web requires the [web] extra: pip install 'openclatura[web]'") from exc

from .. import __version__, describe, name_many
from .. import name as name_one


class NameRequest(BaseModel):
    smiles: str = Field(..., description="A single SMILES string")
    include_trace: bool = Field(False, description="Include trace_segments in response")
    verify_opsin: bool = Field(False, description="Round-trip the generated name via OPSIN")
    token_debug: bool = Field(False, description="Include verbose emitted token metadata in decisions")


class BatchRequest(BaseModel):
    smiles: list[str] = Field(..., description="SMILES strings to name")
    include_trace: bool = False
    verify_opsin: bool = False
    token_debug: bool = False
    processes: int | None = Field(1, description="1=serial, None/null=all CPUs, integer=N workers")
    chunksize: int = 64


class DescribeRequest(BaseModel):
    smiles: str


def create_app() -> FastAPI:
    """Build the FastAPI app. Factory style keeps tests honest."""

    app = FastAPI(
        title="openclatura",
        version=__version__,
        description="SMILES → IUPAC name service (Blue Book rules).",
    )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.post("/name")
    def name_endpoint(req: NameRequest) -> dict[str, Any]:
        result = name_one(
            req.smiles,
            include_trace=req.include_trace,
            verify_opsin=req.verify_opsin,
            token_debug=req.token_debug,
        )
        return result.to_dict(include_trace=req.include_trace)

    @app.post("/batch")
    def batch_endpoint(req: BatchRequest) -> list[dict[str, Any]]:
        results = name_many(
            req.smiles,
            include_trace=req.include_trace,
            verify_opsin=req.verify_opsin,
            token_debug=req.token_debug,
            processes=req.processes,
            chunksize=req.chunksize,
        )
        return [r.to_dict(include_trace=req.include_trace) for r in results]

    @app.post("/describe")
    def describe_endpoint(req: DescribeRequest) -> dict[str, Any]:
        return describe(req.smiles).to_dict()

    return app


app = create_app()
