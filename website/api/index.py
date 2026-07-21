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
import importlib.metadata
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

_JRE_TARBALL = Path(__file__).parent / "jre.tar.gz"
_JRE_DIR = Path(tempfile.gettempdir()) / "openclatura-jre"
_XLIBS_TARBALL = Path(__file__).parent / "xlibs.tar.gz"
_XLIBS_DIR = Path(tempfile.gettempdir()) / "openclatura-xlibs"

# rdkit.Chem.Draw dlopens a bundled libcairo that expects these X11 libs,
# which the Vercel runtime image lacks. Preloading them (dependency order,
# RTLD_GLOBAL) makes the later dlopen resolve against the loaded sonames.
_XLIBS_ORDER = (
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

from openclatura.web.app import create_app  # noqa: E402

app = FastAPI()


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
    return {"ok": True, "svg": svg}


app.mount("/api", create_app())
