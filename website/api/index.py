"""Vercel serverless entrypoint.

Vercel passes the original request path (``/api/...``) to the ASGI app,
so the openclatura FastAPI app is mounted under ``/api``:

- POST /api/name      (supports ``verify_opsin`` for OPSIN round-trip)
- POST /api/batch
- POST /api/describe
- GET  /api/healthz

The Vercel Python runtime ships no Java, which py2opsin needs for OPSIN
verification. A jlink-minimized JRE (java.base, java.xml, java.logging,
java.naming, java.management, java.desktop — the closure OPSIN's log4j
needs) is bundled as ``jre.tar.gz`` and extracted to ``/tmp`` on cold
start; its ``bin`` is prepended to ``PATH`` before openclatura checks
``shutil.which("java")``.
"""

import ctypes
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import urllib.parse
import urllib.request
from pathlib import Path

_JRE_TARBALL = Path(__file__).parent / "jre.tar.gz"
_JRE_DIR = Path(tempfile.gettempdir()) / "openclatura-jre"
_XLIBS_TARBALL = Path(__file__).parent / "xlibs.tar.gz"
_XLIBS_DIR = Path(tempfile.gettempdir()) / "openclatura-xlibs"

# rdkit.Chem.Draw dlopens a bundled libcairo that expects these X11 libs,
# which the Vercel runtime image lacks. Preloading them (dependency order,
# RTLD_GLOBAL) makes the later dlopen resolve against the loaded sonames.
_XLIBS_ORDER = (
    "libexpat.so.1",
    "libmd.so.0",
    "libbsd.so.0",
    "libXau.so.6",
    "libXdmcp.so.6",
    "libxcb.so.1",
    "libX11.so.6",
    "libXext.so.6",
    "libXrender.so.1",
)


def _preload_xlibs() -> None:
    if not _XLIBS_TARBALL.exists():
        return
    if not (_XLIBS_DIR / _XLIBS_ORDER[-1]).exists():
        staging = Path(tempfile.mkdtemp(dir=tempfile.gettempdir()))
        with tarfile.open(_XLIBS_TARBALL) as tar:
            tar.extractall(staging)
        try:
            staging.rename(_XLIBS_DIR)
        except OSError:  # concurrent cold start already extracted it
            shutil.rmtree(staging, ignore_errors=True)
    for soname in _XLIBS_ORDER:
        try:
            ctypes.CDLL(str(_XLIBS_DIR / soname), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # already provided by the system, or depiction stays off


def _ensure_java() -> None:
    if shutil.which("java"):
        return
    if not _JRE_TARBALL.exists():
        return
    if not (_JRE_DIR / "bin" / "java").exists():
        staging = Path(tempfile.mkdtemp(dir=tempfile.gettempdir()))
        with tarfile.open(_JRE_TARBALL) as tar:
            tar.extractall(staging)
        try:
            staging.rename(_JRE_DIR)
        except OSError:  # concurrent cold start already extracted it
            shutil.rmtree(staging, ignore_errors=True)
    os.environ["PATH"] = f"{_JRE_DIR / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.setdefault("JAVA_HOME", str(_JRE_DIR))


_ensure_java()

# py2opsin writes its temp input file relative to the working directory;
# on Vercel only /tmp is writable.
os.chdir(tempfile.gettempdir())

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from rdkit import Chem  # noqa: E402

# Depiction is optional: never let a missing native lib break the whole API.
_preload_xlibs()
try:
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError as exc:
    rdMolDraw2D = None
    _DRAW_IMPORT_ERROR = str(exc)

from openclatura import describe  # noqa: E402
from openclatura import name as name_one  # noqa: E402
from openclatura.web.app import create_app  # noqa: E402

app = FastAPI()

# ---------------------------------------------------------------------------
# Result cache (Upstash Redis over REST). Naming is deterministic per package
# version, so results are cached under the canonical SMILES + request flags,
# scoped by version. Fully optional: without credentials every request just
# computes, and any cache error falls back to computing.
# ---------------------------------------------------------------------------
_CACHE_URL = (os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL") or "").rstrip("/")
_CACHE_TOKEN = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN") or ""
_CACHE_TTL_SECONDS = 90 * 24 * 3600


def _pkg_version() -> str:
    try:
        return importlib.metadata.version("openclatura")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _cache_key(kind: str, smiles: str, flags: str = "") -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:32]
    return f"{kind}:{_pkg_version()}:{flags}:{digest}"


def _cache_get(key: str | None) -> dict | None:
    if not key or not _CACHE_URL:
        return None
    req = urllib.request.Request(
        f"{_CACHE_URL}/get/{urllib.parse.quote(key, safe='')}",
        headers={"Authorization": f"Bearer {_CACHE_TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            raw = json.load(resp).get("result")
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _cache_set(key: str | None, value: dict) -> None:
    if not key or not _CACHE_URL:
        return
    req = urllib.request.Request(
        f"{_CACHE_URL}/set/{urllib.parse.quote(key, safe='')}?EX={_CACHE_TTL_SECONDS}",
        data=json.dumps(value).encode(),
        headers={"Authorization": f"Bearer {_CACHE_TOKEN}"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=2).read()
    except Exception:
        pass

# py2opsin writes a fixed-name temp file in the CWD, so concurrent
# requests in one warm process corrupt each other's OPSIN round-trip.
# Serialize the verify path (and retry once for residual flakes).
_OPSIN_LOCK = threading.Lock()


class NameRequest(BaseModel):
    smiles: str
    include_trace: bool = False
    verify_opsin: bool = False
    token_debug: bool = False


def _name_cacheable(payload: dict, verify: bool) -> bool:
    if not payload.get("ok"):
        return False
    if verify:
        status = (payload.get("opsin_check") or {}).get("status")
        return status in ("matched", "mismatched", "name_unparseable")
    return True


@app.post("/api/name")
def name_endpoint(req: NameRequest) -> dict:
    """Shadows the mounted app's /name: adds result caching and makes
    OPSIN verification safe under in-process request concurrency."""
    flags = f"t{int(req.include_trace)}v{int(req.verify_opsin)}d{int(req.token_debug)}"
    key = _cache_key("name", req.smiles, flags)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    if not req.verify_opsin:
        result = name_one(req.smiles, include_trace=req.include_trace, token_debug=req.token_debug)
    else:
        with _OPSIN_LOCK:
            for attempt in (1, 2):
                result = name_one(
                    req.smiles,
                    include_trace=req.include_trace,
                    verify_opsin=True,
                    token_debug=req.token_debug,
                )
                if result.opsin_check is None or result.opsin_check.status != "error":
                    break
    payload = result.to_dict(include_trace=req.include_trace)
    if _name_cacheable(payload, req.verify_opsin):
        _cache_set(key, payload)
    return payload


class DescribeRequest(BaseModel):
    smiles: str


@app.post("/api/describe")
def describe_endpoint(req: DescribeRequest) -> dict:
    """Shadows the mounted app's /describe to add result caching."""
    key = _cache_key("desc", req.smiles)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    payload = describe(req.smiles).to_dict()
    _cache_set(key, payload)
    return payload


@app.get("/api/healthz")
def healthz() -> dict:
    """Report the installed distribution version.

    Shadows the mounted app's healthz: released wheels up to 0.1.4 ship a
    stale hardcoded ``openclatura.__version__``.
    """
    try:
        version = importlib.metadata.version("openclatura")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"ok": True, "version": version}


class DepictRequest(BaseModel):
    smiles: str
    width: int = Field(440, ge=100, le=1200)
    height: int = Field(360, ge=100, le=1200)


@app.post("/api/depict")
def depict(req: DepictRequest) -> dict:
    """Render the molecule as SVG with RDKit atom indices annotated.

    The indices match the atom ids reported by ``/api/describe``
    (both are plain RDKit atom indices of ``MolFromSmiles(smiles)``).
    """
    if rdMolDraw2D is None:
        return {"ok": False, "error": f"Depiction unavailable: {_DRAW_IMPORT_ERROR}"}
    key = _cache_key("depict", req.smiles, f"{req.width}x{req.height}")
    cached = _cache_get(key)
    if cached is not None:
        return cached
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        return {"ok": False, "error": "RDKit could not parse the SMILES."}
    drawer = rdMolDraw2D.MolDraw2DSVG(req.width, req.height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    opts.annotationFontScale = 0.7
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # Drop the XML declaration so the SVG can be injected via innerHTML.
    svg = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", svg)
    payload = {"ok": True, "svg": svg}
    _cache_set(key, payload)
    return payload


app.mount("/api", create_app())
