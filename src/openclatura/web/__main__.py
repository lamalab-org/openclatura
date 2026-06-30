"""``python -m openclatura.web`` — run the FastAPI app with uvicorn."""

from __future__ import annotations

import argparse


def main() -> int:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - extra not installed
        raise SystemExit("uvicorn is not installed. Install the [web] extra: pip install 'openclatura[web]'") from exc

    parser = argparse.ArgumentParser(prog="openclatura.web", description="Run the openclatura HTTP service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "openclatura.web.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
