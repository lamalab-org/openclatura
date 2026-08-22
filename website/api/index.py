"""Vercel serverless entrypoint.

Vercel passes the original request path (``/api/...``) to the ASGI app,
so the openclatura FastAPI app is mounted under ``/api``:

- POST /api/name      (``verify_opsin`` for the OPSIN round-trip, ``verify_self``
                      for the OPSIN-free reconstruction audit; both may run)
- POST /api/batch
- POST /api/describe
- GET  /api/healthz   (also reports the release channel)

The Vercel Python runtime ships no Java, which py2opsin needs for OPSIN
verification. A jlink-minimized JRE (java.base, java.xml, java.logging,
java.naming, java.management, java.desktop — the closure OPSIN's log4j
needs) is bundled as ``jre.tar.gz`` and extracted to ``/tmp`` on cold
start; its ``bin`` is prepended to ``PATH`` before openclatura checks
``shutil.which("java")``.
"""

import ctypes
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# BigSMILES bridge.
#
# Imported here, before the openclatura and RDKit imports below, on purpose:
# ``bigsmiles`` pulls in ``pkg_resources`` -> ``plistlib`` -> ``pyexpat``, and a
# ``pyexpat`` loaded after RDKit binds against RDKit's own libexpat and fails on
# a missing symbol. Loading it first makes the order deterministic.
#
# Optional either way: the page still names polymers if the import fails, it just
# loses the paste-a-BigSMILES route and the BigSMILES readout.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
try:
    from polymer_bigsmiles import (
        looks_like_bigsmiles,
        read_bigsmiles,
        strip_graft_sites,
        write_bigsmiles,
    )

    _BIGSMILES_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the deployed wheels
    _BIGSMILES_ERROR = str(exc)

    def looks_like_bigsmiles(text: str) -> bool:
        return "{" in (text or "") and "}" in (text or "")

    def read_bigsmiles(text, route_choice=None) -> dict:
        return {"ok": False, "error": f"BigSMILES support unavailable: {_BIGSMILES_ERROR}"}

    def write_bigsmiles(monomers, connective="co", end_alpha=None,
                        end_omega=None, blocks=None) -> dict:
        return {"ok": False, "reason": "BigSMILES support unavailable"}

    def strip_graft_sites(smiles):
        return None


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
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    # CoordGen lays rings out the way a chemist would draw them, which is what
    # makes the depictions look hand-drawn rather than machine-generated.
    rdDepictor.SetPreferCoordGen(True)
except ImportError as exc:
    rdMolDraw2D = None
    _DRAW_IMPORT_ERROR = str(exc)

from openclatura import describe  # noqa: E402
from openclatura import describe_human  # noqa: E402
from openclatura import name as name_one  # noqa: E402
from openclatura.web.app import create_app  # noqa: E402

app = FastAPI()

# ---------------------------------------------------------------------------
# Release channel.
#
# The openclatura version is fixed at build time by requirements.txt, so one
# deployment serves exactly one version. Two Vercel projects share this code:
#
#   stable  CHANNEL unset (or "stable"), requirements.txt pins a PyPI release.
#   beta    CHANNEL=beta, requirements.txt installs from a git ref, and the
#           deployment lives on its own separate domain.
#
# There is no authentication on beta. Its only protection is that the domain is
# not advertised: leave BETA_URL unset on stable and the public site never links
# to or names it. Anyone given the URL can use it.
#
# STABLE_URL lets beta offer the switch back, and BETA_URL (if you do set it)
# lets stable offer the switch forward. BETA_GIT_REF is display-only: a git
# install reports the same version string as the release it branched from, so
# the ref is what actually tells you which build you are looking at.
# ---------------------------------------------------------------------------
_CHANNEL = (os.environ.get("CHANNEL") or "stable").strip().lower()
_IS_BETA = _CHANNEL == "beta"
_BETA_URL = (os.environ.get("BETA_URL") or "").rstrip("/")
_STABLE_URL = (os.environ.get("STABLE_URL") or "").rstrip("/")
_BETA_GIT_REF = os.environ.get("BETA_GIT_REF") or ""

# ---------------------------------------------------------------------------
# Result cache. Naming is deterministic per package version, so results are
# cached under the canonical SMILES + request flags, scoped by version.
# Backends, first configured wins:
#   1. S3-compatible object store (CACHE_S3_*; AWS_* accepted locally, but the
#      AWS_* names are reserved on Vercel where Lambda injects its own).
#      Expiry comes from a bucket lifecycle rule, not per-request TTL.
#   2. Upstash Redis over REST (KV_REST_API_* / UPSTASH_REDIS_REST_*).
# Fully optional: without credentials every request just computes, and any
# cache error falls back to computing.
#
# Requests may set ``no_cache`` to opt out: their structure is then never
# written to the cache. Reads still happen (serving an already-cached result
# stores nothing new, and keeps the response fast).
# ---------------------------------------------------------------------------
_CACHE_URL = (os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL") or "").rstrip("/")
_CACHE_TOKEN = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN") or ""
_CACHE_TTL_SECONDS = 90 * 24 * 3600

_S3_ENDPOINT = (os.environ.get("CACHE_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or "").rstrip("/")
_S3_BUCKET = os.environ.get("CACHE_S3_BUCKET", "")
_S3_ACCESS = os.environ.get("CACHE_S3_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID") or ""
_S3_SECRET = os.environ.get("CACHE_S3_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY") or ""
_S3_REGION = os.environ.get("CACHE_S3_REGION", "us-east-1")
_S3_ENABLED = bool(_S3_ENDPOINT and _S3_BUCKET and _S3_ACCESS and _S3_SECRET)


def _s3_request(method: str, key: str, body: bytes = b"", timeout: float = 2.0) -> tuple[int, bytes]:
    """Minimal SigV4-signed path-style S3 request (stdlib only)."""
    import datetime
    import hmac

    path = f"/{_S3_BUCKET}/{key}"
    host = urllib.parse.urlparse(_S3_ENDPOINT).netloc
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body).hexdigest()
    headers = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
    signed = ";".join(sorted(headers))
    canonical = "\n".join(
        [
            method,
            urllib.parse.quote(path),
            "",
            "".join(f"{k}:{headers[k]}\n" for k in sorted(headers)),
            signed,
            payload_hash,
        ]
    )
    scope = f"{datestamp}/{_S3_REGION}/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k = f"AWS4{_S3_SECRET}".encode()
    for part in (datestamp, _S3_REGION, "s3", "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(
        _S3_ENDPOINT + urllib.parse.quote(path),
        data=body if method == "PUT" else None,
        method=method,
        headers={
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
            "Authorization": f"AWS4-HMAC-SHA256 Credential={_S3_ACCESS}/{scope}, "
            f"SignedHeaders={signed}, Signature={signature}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except Exception:
        return 0, b""


def _pkg_version() -> str:
    try:
        return importlib.metadata.version("openclatura")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _cache_key(kind: str, smiles: str, flags: str = "") -> str | None:
    # The beta channel never caches. Keys are scoped by package version, but a
    # git install reports the version of the release its branch forked from, so
    # beta would both collide with stable's entries and pin results to whichever
    # commit of the branch was named first. Returning None disables get and set;
    # beta traffic is a handful of testers, so there is nothing to gain anyway.
    if _IS_BETA:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:32]
    return f"{kind}/{_pkg_version()}/{flags or '-'}/{digest}.json"


def _cache_get(key: str | None) -> dict | None:
    if not key:
        return None
    if _S3_ENABLED:
        status, body = _s3_request("GET", key)
        if status != 200:
            return None
        try:
            return json.loads(body)
        except Exception:
            return None
    if not _CACHE_URL:
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
    if not key:
        return
    if _S3_ENABLED:
        _s3_request("PUT", key, body=json.dumps(value).encode(), timeout=3.0)
        return
    if not _CACHE_URL:
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


class _OpsinDaemon:
    """Long-lived OPSIN CLI process, one per instance.

    Spawning a JVM per verification costs ~1.5 s; OPSIN's CLI streams
    names line-by-line (empty output line = unparseable), so one warm
    JVM answers in milliseconds. Access is serialized by _OPSIN_LOCK.
    On any protocol hiccup the process is killed and the caller falls
    back to the one-shot py2opsin path.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    @staticmethod
    def _jar_path() -> str | None:
        try:
            import py2opsin
        except ImportError:
            return None
        jars = sorted(Path(py2opsin.__file__).parent.glob("opsin-cli-*.jar"))
        return str(jars[-1]) if jars else None

    def _ensure(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        jar = self._jar_path()
        if jar is None or not shutil.which("java"):
            return False
        self._proc = subprocess.Popen(
            ["java", "-jar", jar, "-osmi"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        return True

    def parse(self, name: str, timeout: float = 15.0) -> str | None:
        """SMILES for ``name``, "" if OPSIN can't parse it, None if the
        daemon is unavailable (caller should fall back)."""
        if "\n" in name or "\r" in name or not self._ensure():
            return None
        proc = self._proc
        try:
            proc.stdin.write(name + "\n")
            proc.stdin.flush()
            ready, _, _ = select.select([proc.stdout], [], [], timeout)
            if not ready:
                raise TimeoutError(f"OPSIN gave no answer within {timeout}s")
            line = proc.stdout.readline()
            if line == "":
                raise EOFError("OPSIN process closed stdout")
            return line.rstrip("\n")
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            self._proc = None
            return None


_OPSIN_DAEMON = _OpsinDaemon()


def _opsin_check_via_daemon(name: str, smiles: str):
    """Mirror openclatura.opsin_verify.verify_with_opsin, but decode the
    name through the persistent JVM. Returns None to request fallback."""
    from openclatura.opsin_verify import OpsinCheck
    from openclatura.resonance_compare import canonical_smiles, equivalent_smiles
    from openclatura.utils import standardize_mol

    if not name:
        return OpsinCheck(status="name_empty", name=name)
    decoded = _OPSIN_DAEMON.parse(name)
    if decoded is None:
        return None
    canonical_original = standardize_mol(smiles)
    if decoded == "":
        return OpsinCheck(status="name_unparseable", name=name, canonical_original=canonical_original)
    canonical_roundtrip = canonical_smiles(decoded)
    if canonical_original is None or canonical_roundtrip is None:
        return OpsinCheck(
            status="error",
            name=name,
            canonical_original=canonical_original,
            opsin_smiles=decoded,
            canonical_roundtrip=canonical_roundtrip,
            error_message="Failed to standardize SMILES for comparison.",
        )
    return OpsinCheck(
        status="matched" if equivalent_smiles(smiles, decoded) else "mismatched",
        name=name,
        canonical_original=canonical_original,
        opsin_smiles=decoded,
        canonical_roundtrip=canonical_roundtrip,
    )


# The OPSIN-free reconstruction audit (``verify_self``) only exists on branches
# that ship openclatura.audit — currently the beta channel. Feature-detect rather
# than key off CHANNEL, so passing the flag can never raise TypeError on stable.
_SELF_AUDIT_AVAILABLE = "verify_self" in inspect.signature(name_one).parameters


class NameRequest(BaseModel):
    smiles: str
    include_trace: bool = False
    verify_opsin: bool = False
    # Rebuild the molecule from its own name using openclatura's grammar and
    # compare (no Java, no OPSIN). Ignored where the engine lacks support.
    verify_self: bool = False
    token_debug: bool = False
    # Opt out of having this structure written to the result cache.
    no_cache: bool = False


def _name_cacheable(payload: dict, verify: bool) -> bool:
    if not payload.get("ok"):
        return False
    # A self-audit that blew up is a transient result, not a verdict to keep.
    if (payload.get("self_audit") or {}).get("verdict") == "error":
        return False
    if verify:
        status = (payload.get("opsin_check") or {}).get("status")
        return status in ("matched", "mismatched", "name_unparseable")
    return True


@app.post("/api/name")
def name_endpoint(req: NameRequest) -> dict:
    """Shadows the mounted app's /name: adds result caching and makes
    OPSIN verification safe under in-process request concurrency.

    Both checks can run on one request. They are independent: OPSIN parses the
    name with a third-party parser, the self-audit rebuilds it with our own
    grammar, so agreement between them is worth more than either alone.
    """
    want_self = req.verify_self and _SELF_AUDIT_AVAILABLE
    # The self-audit hooks component naming from the inside, so it has to be
    # requested on the naming call itself — it cannot be bolted on afterwards
    # the way the OPSIN round-trip can.
    self_kwargs = {"verify_self": True} if want_self else {}
    flags = f"t{int(req.include_trace)}v{int(req.verify_opsin)}s{int(want_self)}d{int(req.token_debug)}"
    key = _cache_key("name", req.smiles, flags)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    if not req.verify_opsin:
        result = name_one(
            req.smiles,
            include_trace=req.include_trace,
            token_debug=req.token_debug,
            **self_kwargs,
        )
    else:
        with _OPSIN_LOCK:
            # include_trace=True mirrors the engine's verify branch, which
            # always analyzes; to_dict() below trims it when not requested.
            result = name_one(req.smiles, include_trace=True, token_debug=req.token_debug, **self_kwargs)
            check = None if result.error else _opsin_check_via_daemon(result.name, req.smiles)
            if check is not None:
                result = dataclasses.replace(result, opsin_check=check)
            else:
                for attempt in (1, 2):
                    result = name_one(
                        req.smiles,
                        include_trace=req.include_trace,
                        verify_opsin=True,
                        token_debug=req.token_debug,
                        **self_kwargs,
                    )
                    if result.opsin_check is None or result.opsin_check.status != "error":
                        break
    payload = result.to_dict(include_trace=req.include_trace)
    if not req.no_cache and _name_cacheable(payload, req.verify_opsin):
        _cache_set(key, payload)
    return payload


class DescribeRequest(BaseModel):
    smiles: str
    no_cache: bool = False


@app.post("/api/describe")
def describe_endpoint(req: DescribeRequest) -> dict:
    """Shadows the mounted app's /describe to add result caching."""
    key = _cache_key("desc", req.smiles)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    payload = describe(req.smiles).to_dict()
    if not req.no_cache:
        _cache_set(key, payload)
    return payload


@app.get("/api/warmup")
def warmup() -> dict:
    """Boot the OPSIN JVM and prime the naming engine.

    The frontend calls this fire-and-forget while the Ketcher editor is
    still loading, so the cold start and JVM boot overlap with editor
    init instead of delaying the first real naming request.
    """
    t0 = time.time()
    with _OPSIN_LOCK:
        opsin_ready = _OPSIN_DAEMON.parse("methane") is not None
    try:
        name_one("C")
        engine_ready = True
    except Exception:
        engine_ready = False
    return {"ok": True, "opsin": opsin_ready, "engine": engine_ready, "seconds": round(time.time() - t0, 2)}


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
    return {
        "ok": True,
        "version": version,
        "channel": _CHANNEL,
        "git_ref": _BETA_GIT_REF,
        "self_audit": _SELF_AUDIT_AVAILABLE,
        "beta_url": _BETA_URL,
        "stable_url": _STABLE_URL,
    }


class DepictRequest(BaseModel):
    smiles: str
    width: int = Field(440, ge=100, le=1200)
    height: int = Field(360, ge=100, le=1200)
    # Atom indices are what makes this useful next to /api/describe, but they are
    # noise in a small thumbnail, so callers that only want the structure
    # (the polymer builder's monomer cards) can turn them off.
    annotate: bool = True
    no_cache: bool = False


@app.post("/api/depict")
def depict(req: DepictRequest) -> dict:
    """Render the molecule as SVG, optionally with RDKit atom indices annotated.

    The indices match the atom ids reported by ``/api/describe``
    (both are plain RDKit atom indices of ``MolFromSmiles(smiles)``).
    """
    if rdMolDraw2D is None:
        return {"ok": False, "error": f"Depiction unavailable: {_DRAW_IMPORT_ERROR}"}
    key = _cache_key("depict", req.smiles, f"{req.width}x{req.height}a{int(req.annotate)}")
    cached = _cache_get(key)
    if cached is not None:
        return cached
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        return {"ok": False, "error": "RDKit could not parse the SMILES."}
    drawer = rdMolDraw2D.MolDraw2DSVG(req.width, req.height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = req.annotate
    opts.annotationFontScale = 0.7
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # Drop the XML declaration so the SVG can be injected via innerHTML.
    svg = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", svg)
    payload = {"ok": True, "svg": svg}
    if not req.no_cache:
        _cache_set(key, payload)
    return payload


# ---------------------------------------------------------------------------
# /api/teach — everything the teaching page needs, in one round trip.
#
# It differs from /api/describe in what it is allowed to say. The teaching page
# explains a name to someone learning nomenclature, so nothing about how the
# software works may leak into it: no SMILES, no RDKit, no atom indices, no
# verification, no caching. Positions are IUPAC locants, and the structure is
# drawn with those locants rather than the arbitrary input numbering.
# ---------------------------------------------------------------------------

# What each name token is doing, in the reader's terms rather than the
# grammar's. Unlisted kinds fall back to no gloss at all rather than to a
# vague one.
_TOKEN_ROLES = {
    "parent": "the parent hydride — the skeleton the rest of the name hangs off",
    "suffix": "the ending for the principal characteristic group",
    "prefix": "a substituent named as a prefix",
    "replacement": "a replacement prefix, naming an atom that stands in for carbon",
    "locant": "a locant — it says which numbered position the next piece sits on",
    "multiplier": "a multiplying prefix, counting identical pieces",
    "hydro": "added hydrogen on the skeleton",
    "unsaturation": "a double or triple bond in the skeleton",
    "stereo": "a stereodescriptor, fixing the three-dimensional arrangement",
    "charge": "an electric charge carried by the skeleton",
}

# Kinds not listed above are the engine's catch-alls, and they differ between
# releases, so they get a gloss that is true of every token rather than a guess
# at the grammar. The highlight still shows what the piece refers to. Every role
# is a noun phrase: the page renders them as `"<token>" is <role>.`
_GENERIC_ROLE = "the part of the name that describes the highlighted atoms"

# What a run of tokens that all resolved to the same atoms gets merged into.
# See _merge_unresolved.
_MERGED_ROLE = "the part of the name that names the highlighted group as a whole"

# Multiplying prefixes arrive tagged "grammar", alongside the commas and
# brackets, so the one piece of that group worth explaining is matched by text.
_MULTIPLIERS = {
    "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "nona", "deca",
    "undeca", "dodeca", "bis", "tris", "tetrakis", "pentakis", "hexakis",
}
_MULTIPLIER_ROLE = "a multiplying prefix, counting how many identical pieces there are"

# Machine detail describe_human() interleaves with its prose. The locants are
# the teaching content; the raw atom ids are the implementation showing through.
_ATOM_ID_NOISE = re.compile(r"\s*\(atom ids?\s*[\d,\s]+\)")


def _iupac_locants(tree: list) -> dict[int, str]:
    """Map atom id -> IUPAC locant for the parent of each top-level component.

    Only the parents get numbered. Substituents carry their own local numbering
    that would collide with the parent's on a single drawing, and it is the
    parent numbering that the locants in the name refer to.
    """
    locants: dict[int, str] = {}
    for node in tree or []:
        if not isinstance(node, dict):
            continue
        parent = node.get("parent")
        if not isinstance(parent, dict):
            continue
        for locant, atom_id in (parent.get("atom_ids_by_locant") or {}).items():
            if isinstance(atom_id, int) and atom_id not in locants:
                locants[atom_id] = str(locant)
    return locants


def _enclosed_bonds(mol, atoms: list[int]) -> list[int]:
    """RDKit bond indices with both ends inside ``atoms``.

    The spans carry bond ids too, but they are openclatura's own: ``add_bond``
    auto-numbers from 1 while RDKit numbers from 0, so using them directly
    highlights the wrong bond (the "amide" of 3-methylbutanamide lit up the
    methyl branch). Atom ids do agree between the two, and re-deriving the
    bonds from them keeps the drawing independent of either numbering.
    """
    if mol is None:
        return []
    inside = set(atoms)
    return [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in inside and bond.GetEndAtomIdx() in inside
    ]


def _teach_tokens(name: str, spans: list, mol) -> list[dict]:
    """Trim the token spans to the ones that genuinely index into ``name``.

    Three filters, because the page lays the tokens out along the name itself:

    - Span offsets are assigned per named component, so on a multi-component
      name they need not line up with the assembled string. A span whose slice
      doesn't reproduce its own text would highlight the wrong letters.
    - Pure punctuation (the commas between locants) is name grammar, not a
      piece to hover.
    - Stems are emitted at several lengths for the same parent ("but", "buta",
      "butan"). Longest-first greedy keeps the fullest form and leaves the
      remaining tokens non-overlapping, which is what a linear layout needs.
    """
    candidates = []
    for span in spans or []:
        if not isinstance(span, dict):
            continue
        start, end, text = span.get("start"), span.get("end"), span.get("text")
        if not isinstance(start, int) or not isinstance(end, int) or not text:
            continue
        if name[start:end] != text or not any(ch.isalnum() for ch in text):
            continue
        kind = str(span.get("token_kind") or "")
        role = _TOKEN_ROLES.get(kind, "")
        if not role and text.lower() in _MULTIPLIERS:
            role = _MULTIPLIER_ROLE
        if not role:
            role = _GENERIC_ROLE
        atoms = [a for a in span.get("atoms") or [] if isinstance(a, int)]
        candidates.append(
            {
                "text": text,
                "start": start,
                "end": end,
                "kind": kind,
                "role": role,
                "atoms": atoms,
                "bonds": _enclosed_bonds(mol, atoms),
                "locants": [str(locant) for locant in span.get("locants") or []],
            }
        )

    tokens: list[dict] = []
    taken: set[int] = set()
    for token in sorted(candidates, key=lambda t: (t["start"] - t["end"], t["start"])):
        positions = set(range(token["start"], token["end"]))
        if positions & taken:
            continue
        taken |= positions
        tokens.append(token)
    tokens.sort(key=lambda t: (t["start"], t["end"]))
    return _narrow_scope_heads(_merge_unresolved(name, tokens), mol)


def _narrow_scope_heads(tokens: list[dict], mol) -> list[dict]:
    """Trim a catch-all token down to the atoms its own word denotes.

    A token kind outside the known grammar is the engine saying it could not
    place the word, and its fallback is to bind the whole enclosing scope. So in
    "1-(3-((2-ethylphenyl)amino)phenyl)ethan-1-one" the final "phenyl" came back
    owning all fifteen atoms of the substituent — both rings, the NH and the
    ethyl — when it names only the ring the rest hangs off.

    That word is the head of the substituent name, and a head denotes what is
    left once its modifiers are accounted for, so subtracting the atoms of the
    tokens nested inside it recovers exactly the ring.

    Restricted to catch-all kinds on purpose: a parent legitimately contains its
    own suffix ("benzoic" spans the ring *and* the acid carbon it shares with
    "acid"), and subtracting there would eat the attachment atom.
    """
    for token in tokens:
        if token["role"] != _GENERIC_ROLE:
            continue
        own = set(token["atoms"])
        if not own:
            continue
        nested: set[int] = set()
        for other in tokens:
            # A locant points at a position rather than naming a constituent, so
            # its atoms are not a part to subtract. They are also the least
            # reliable bindings: the "2" of "2-ethylphenyl" here comes back
            # owning a carbon of the *other* ring, which would otherwise punch a
            # hole in the ring this head denotes.
            if other is token or other["kind"] == "locant":
                continue
            other_atoms = set(other["atoms"])
            if other_atoms and other_atoms < own:
                nested |= other_atoms
        remaining = own - nested
        # Everything subtracted means the word adds nothing of its own; keep the
        # original binding rather than highlight nothing at all.
        if remaining and remaining != own:
            token["atoms"] = sorted(remaining)
            token["bonds"] = _enclosed_bonds(mol, token["atoms"])
    return tokens


def _merge_unresolved(name: str, tokens: list[dict]) -> list[dict]:
    """Collapse neighbouring tokens that all resolved to the same atoms.

    When the engine cannot bind a substituent's words individually it falls back
    to one binding covering the whole scope, then splits that into a token per
    word — so in "2-((cyclopropyl)carbonyl)phenyl acetate" each of "2",
    "cyclopropyl", "carbonyl" and "phenyl" came back owning the entire
    substituent. Highlighting them separately tells the reader "phenyl means all
    of this", which is false, and four identical highlights teach nothing
    besides. Merged into one piece spanning the combined text, the claim is true
    again: that stretch of the name does name that group.

    The run must also agree on ``kind``, which is what separates the fallback
    from the many legitimate cases of neighbours sharing atoms: a locant and the
    prefix it positions both own the methyl carbon in "3-methyl", as do "tri"
    and "methyl", or "2,6" and "dione" — different roles pointing at one group,
    each correctly bound, and each worth hovering on its own.
    """
    merged: list[dict] = []
    run: list[dict] = []

    def flush() -> None:
        if not run:
            return
        if len(run) == 1:
            merged.append(run[0])
        else:
            start, end = run[0]["start"], run[-1]["end"]
            merged.append(
                {
                    "text": name[start:end],
                    "start": start,
                    "end": end,
                    "kind": "group",
                    "role": _MERGED_ROLE,
                    "atoms": run[0]["atoms"],
                    "bonds": run[0]["bonds"],
                    "locants": [],
                }
            )
        run.clear()

    for token in tokens:
        joins = bool(run) and token["atoms"] == run[-1]["atoms"] and token["kind"] == run[-1]["kind"]
        if not joins:
            flush()
        run.append(token)
    flush()
    return merged


def _teach_steps(smiles: str) -> list[str]:
    """Plain-language sentences about how the name is built.

    ``describe_human`` already renders the substituent tree as prose; its first
    paragraph is the SMILES echo, which is exactly what this page must not show.
    """
    human = describe_human(smiles)
    steps: list[str] = []
    for paragraph in human.paragraphs[1:]:
        if paragraph.startswith(("Input SMILES", "Processed SMILES")):
            continue
        for line in paragraph.splitlines():
            line = _ATOM_ID_NOISE.sub("", line).strip()
            # The "named X" line duplicates the name shown above it.
            if line and not line.startswith("The molecule is named "):
                steps.append(line)
    return steps


# Drawing style, ported from AdrianM0/smiles-hover so structures here look like
# the ones that extension renders. Two ideas do most of the work:
#
#  - the canvas is sized from the layout at a fixed number of pixels per bond,
#    rather than being a fixed box the structure is stretched to fill. RDKit
#    fits the molecule to whatever canvas it is given and scales bond width with
#    it, so the canvas *is* the drawing scale; holding it fixed keeps line
#    weight and label size in proportion from ethanol up to a fused polycycle.
#  - the result is then cropped to the ink, because RDKit preserves aspect and
#    leaves the remainder as blank bands.
#
# Atom colours follow the extension's "chemdraw" preset: muted, print-like.
_CHEMDRAW_PALETTE = {
    1: (0.4, 0.4, 0.4),  # H
    5: (0.9, 0.55, 0.3),  # B
    6: (0.1, 0.1, 0.1),  # C
    7: (0.15, 0.3, 0.85),  # N
    8: (0.85, 0.15, 0.15),  # O
    9: (0.2, 0.65, 0.3),  # F
    15: (0.95, 0.55, 0.1),  # P
    16: (0.85, 0.7, 0.1),  # S
    17: (0.2, 0.65, 0.3),  # Cl
    35: (0.55, 0.2, 0.1),  # Br
    53: (0.55, 0.1, 0.65),  # I
}

_PX_PER_BOND = 96  # a bond is one unit long; tuned on caffeine
_MIN_SIDE = 0.9  # so a flat or single-atom layout isn't drawn as a sliver
_CANVAS_BOUNDS = (120, 90, 820, 620)  # min w, min h, max w, max h


def _canvas_for(mol, base_width: int, base_height: int) -> tuple[int, int]:
    """Size the canvas to the 2D layout, at a fixed scale."""
    conformer = mol.GetConformer()
    xs = [conformer.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conformer.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    if not xs:
        return base_width, base_height
    # The requested width/height stay meaningful: they scale the whole drawing.
    unit = _PX_PER_BOND * ((base_width * base_height) / (500 * 350)) ** 0.5
    min_w, min_h, max_w, max_h = _CANVAS_BOUNDS
    # No margin added: RDKit fills whatever it is given, so extra room only
    # magnifies the drawing again. Its own `padding` reserves the edge.
    side = lambda extent, lo, hi: int(round(min(hi, max(lo, max(extent, _MIN_SIDE) * unit))))  # noqa: E731
    return side(max(xs) - min(xs), min_w, max_w), side(max(ys) - min(ys), min_h, max_h)


def _crop_to_ink(svg: str, margin: int = 12) -> tuple[str, float, float]:
    """Move the viewBox onto the drawn extent; return it with the crop offset.

    Only the viewBox moves, so the drawing is untouched and the atom draw
    coordinates stay valid — the page's highlight overlay lives inside the same
    SVG and is translated along with everything else.
    """
    xs: list[float] = []
    ys: list[float] = []
    # Drawn elements only: the background rect spans the whole canvas.
    for element in re.findall(r"<(?:path|ellipse|line|text)[^>]*>", svg):
        for x, y in re.findall(r"(-?\d+\.?\d*)[, ](-?\d+\.?\d*)", element):
            xs.append(float(x))
            ys.append(float(y))
    if not xs or max(xs) <= min(xs) or max(ys) <= min(ys):
        return svg, 0.0, 0.0

    x0 = max(0.0, min(xs) - margin)
    y0 = max(0.0, min(ys) - margin)
    w = round(max(xs) - x0 + margin)
    h = round(max(ys) - y0 + margin)
    svg = re.sub(r"\bwidth='[\d.]+px'", f"width='{w}px'", svg, count=1)
    svg = re.sub(r"\bheight='[\d.]+px'", f"height='{h}px'", svg, count=1)
    svg = re.sub(r"\bviewBox='[^']*'", f"viewBox='{x0:.1f} {y0:.1f} {w} {h}'", svg, count=1)
    # The background rect covers the original canvas, which a tight drawing can
    # spill past — leaving the margin transparent. Move it onto the crop.
    svg = re.sub(
        r"(<rect[^>]*?)width='[\d.]+' height='[\d.]+' x='[-\d.]+' y='[-\d.]+'",
        rf"\g<1>width='{w}' height='{h}' x='{x0:.1f}' y='{y0:.1f}'",
        svg,
        count=1,
    )
    return svg, x0, y0


def _teach_depiction(smiles: str, locants: dict[int, str], width: int, height: int) -> dict:
    """Draw the structure annotated with IUPAC locants, plus atom coordinates.

    The coordinates let the page overlay its own highlight markers: RDKit emits
    ``atom-N``/``bond-N`` classes, but an unlabelled carbon has no glyph to
    recolour, so hovering a name token could not light it up otherwise.
    """
    if rdMolDraw2D is None:
        return {"ok": False, "error": f"Depiction unavailable: {_DRAW_IMPORT_ERROR}"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False, "error": "The structure could not be read."}

    # Lay the molecule out first: the layout is what the canvas is sized from.
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    # Preparation only ever appends atoms, so the ids the name binds to survive.
    for atom_id, locant in locants.items():
        if atom_id < mol.GetNumAtoms():
            mol.GetAtomWithIdx(atom_id).SetProp("atomNote", locant)

    canvas_width, canvas_height = _canvas_for(mol, width, height)
    drawer = rdMolDraw2D.MolDraw2DSVG(canvas_width, canvas_height)
    opts = drawer.drawOptions()
    opts.explicitMethyl = True
    opts.multipleBondOffset = 0.18
    # Bolder than the print defaults, so the structure holds up on screen.
    opts.bondLineWidth = 2
    opts.scaleBondWidth = True
    # No fixed font size: it collides labels on anything dense. Scale instead,
    # with a floor that stays readable.
    opts.minFontSize = 12
    opts.maxFontSize = 22
    opts.annotationFontScale = 0.75
    opts.additionalAtomLabelPadding = 0.1
    opts.padding = 0.1
    opts.clearBackground = True
    opts.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    opts.updateAtomPalette(_CHEMDRAW_PALETTE)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    coords = {}
    for atom in mol.GetAtoms():
        point = drawer.GetDrawCoords(atom.GetIdx())
        coords[str(atom.GetIdx())] = [round(point.x, 2), round(point.y, 2)]

    svg = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", drawer.GetDrawingText())
    svg, _, _ = _crop_to_ink(svg)
    return {"ok": True, "svg": svg, "coords": coords, "bond_px": _drawn_bond_length(mol, coords)}


def _drawn_bond_length(mol, coords: dict[str, list[float]]) -> float:
    """Median bond length in drawing pixels.

    The canvas is sized per molecule, so it is the only scale the page can size
    its highlight markers against — a fixed radius that suits caffeine swallows
    ethanol whole.
    """
    lengths = []
    for bond in mol.GetBonds():
        a = coords.get(str(bond.GetBeginAtomIdx()))
        b = coords.get(str(bond.GetEndAtomIdx()))
        if a and b:
            lengths.append(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
    if not lengths:
        return float(_PX_PER_BOND)
    lengths.sort()
    return round(lengths[len(lengths) // 2], 2)


class TeachRequest(BaseModel):
    smiles: str
    width: int = Field(480, ge=100, le=1200)
    height: int = Field(380, ge=100, le=1200)
    no_cache: bool = False


@app.post("/api/teach")
def teach(req: TeachRequest) -> dict:
    """Name a structure and explain the name, for openclatura.org/teach."""
    key = _cache_key("teach", req.smiles, f"{req.width}x{req.height}")
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # describe() rather than name(): the token spans are scattered through the
    # trace and the substituent tree, and the describer is what collects them
    # into one flat, de-duplicated, name-ordered list.
    payload = describe(req.smiles, token_debug=True).to_dict()
    name = payload.get("name")
    if not name:
        # Deliberately not surfacing the engine's error text: it is written for
        # developers, and this page has no developer audience.
        return {"ok": False, "error": "This structure could not be named yet."}

    locants = _iupac_locants(payload.get("substituent_tree") or [])
    tokens = _teach_tokens(name, payload.get("token_spans") or [], Chem.MolFromSmiles(req.smiles))
    out = {
        "ok": True,
        "name": name,
        "tokens": tokens,
        "steps": _teach_steps(req.smiles),
        "depiction": _teach_depiction(req.smiles, locants, req.width, req.height),
        "locants": {str(atom): locant for atom, locant in locants.items()},
    }
    if not req.no_cache:
        _cache_set(key, out)
    return out


# ===========================================================================
# Source-based polymer nomenclature, for openclatura.org/polymers.
#
# Only source-based names are generated here: "poly" applied to the names of the
# monomers the polymer is made from. That maps cleanly onto what the backbone
# already does well -- monomers are ordinary small molecules -- and needs no new
# naming capability.
#
# Structure-based names (poly(oxyethane-1,2-diyl) and friends) are deliberately
# NOT generated. They require naming the repeat unit as a bivalent group, and
# the backbone has no entry point for that: a fragment with attachment points
# names as the empty string, and naive H-capping picks the wrong parent (cap
# -CH2-CH(Ph)- and parent selection lands on the ring, giving ethylbenzene).
# Where a standard structure-based name exists it is *curated* in the commodity
# table below and checked against OPSIN, never derived.
# ===========================================================================

# Copolymer connectives. "block" and "graft" join whole poly(...) blocks; the
# rest join monomer names inside a single poly(...).
_CONNECTIVES = {
    "co": ("unspecified", False),
    "stat": ("statistical", False),
    "ran": ("random (Bernoullian)", False),
    "alt": ("alternating", False),
    "grad": ("gradient", False),
    "per": ("periodic", False),
    "block": ("block", True),
    "graft": ("graft", True),
    # A blend is not a copolymer, but the joiner grammar is the same: whole
    # poly(...) names joined by the italicised connective. Used when the
    # notation asserts separate chains (dot-separated components, or repeat
    # units whose descriptors never bond each other).
    "blend": ("blend (macromolecular mixture)", True),
}

# Configurational qualifiers, cited as italic prefixes before "poly": read off
# stereodescriptors in an imported repeat unit ([C@@H] -> it-), or chosen in
# the builder. They are polymer-level facts the monomer cannot carry --
# propene is flat; isotactic polypropylene is not.
_TACTICITIES = {
    "it": "isotactic",
    "st": "syndiotactic",
    "at": "atactic",
    "cis": "cis double bonds throughout",
    "trans": "trans double bonds throughout",
}

# Architecture qualifiers. "linear" is the default and is left unmarked.
_ARCHITECTURES = {
    "linear": "",
    "cyclo": "cyclo",
    "branch": "branch",
    "star": "star",
    "comb": "comb",
    "net": "net",
    "blend": "blend",
}

# End groups, named as substituent prefixes for the alpha/omega positions.
#
# A curated set rather than derived names: the backbone's substituent mode is
# monovalent-capable but not exposed as a "name this fragment" call. The set is
# organised by where an end group actually comes from -- an initiator residue, a
# controlled-radical chain end, an unreacted condensation handle, or a group put
# there on purpose afterwards -- because that is what decides whether a given end
# group is plausible on a given polymer.
#
# Keys are the label chemists ask for (often a trivial name); values are
# (IUPAC substituent prefix, provenance). Every prefix here round-trips through
# OPSIN to the intended structure.
_ORIGIN_UNREACTIVE = "unreactive chain end"
_ORIGIN_INITIATOR = "radical initiator residue"
_ORIGIN_CONTROLLED = "controlled-radical chain end"
_ORIGIN_HANDLE = "condensation handle"
_ORIGIN_FUNCTIONAL = "installed on purpose"
_ORIGIN_IONIC = "ionic / inorganic end"

_END_GROUPS = {
    # --- plain ends, from transfer or a simple alkyl initiator ---
    "hydro": ("hydro", _ORIGIN_UNREACTIVE),
    "methyl": ("methyl", _ORIGIN_UNREACTIVE),
    "ethyl": ("ethyl", _ORIGIN_UNREACTIVE),
    "propyl": ("propyl", _ORIGIN_UNREACTIVE),
    "butyl": ("butyl", "n-butyllithium initiator"),
    "sec-butyl": ("sec-butyl", "sec-butyllithium initiator"),
    "tert-butyl": ("tert-butyl", _ORIGIN_UNREACTIVE),
    "phenyl": ("phenyl", _ORIGIN_UNREACTIVE),
    "benzyl": ("benzyl", _ORIGIN_UNREACTIVE),
    # --- radical initiator residues ---
    "cyanoisopropyl (AIBN)": ("2-cyanopropan-2-yl", "AIBN initiator residue"),
    "benzoyloxy": ("benzoyloxy", "benzoyl peroxide initiator residue"),
    "tert-butoxy": ("tert-butoxy", _ORIGIN_INITIATOR),
    "sulfooxy": ("sulfooxy", "persulfate initiator residue"),
    # --- controlled radical chain ends ---
    "chloro": ("chloro", "ATRP chain end / halide"),
    "bromo": ("bromo", "ATRP chain end / halide"),
    "TEMPO": ("(2,2,6,6-tetramethylpiperidin-1-yl)oxy", "nitroxide (NMP) chain end"),
    "RAFT dithiobenzoate": ("phenylcarbonothioylsulfanyl", _ORIGIN_CONTROLLED),
    "RAFT trithiocarbonate": ("dodecylsulfanylcarbonothioylsulfanyl", _ORIGIN_CONTROLLED),
    # --- condensation handles, reacted or left over ---
    "hydroxy": ("hydroxy", _ORIGIN_HANDLE),
    "carboxy": ("carboxy", _ORIGIN_HANDLE),
    "amino": ("amino", _ORIGIN_HANDLE),
    "isocyanato": ("isocyanato", "unreacted isocyanate (prepolymer)"),
    "chlorocarbonyl": ("chlorocarbonyl", "unreacted acyl chloride"),
    "formyl": ("formyl", _ORIGIN_HANDLE),
    "acetyl": ("acetyl", "capped by acetylation"),
    "acetoxy": ("acetoxy", "capped by acetylation"),
    "methoxy": ("methoxy", "capped alkoxide / initiator"),
    "ethoxy": ("ethoxy", "capped alkoxide / initiator"),
    # --- groups installed on purpose (click, bioconjugation, cross-linking) ---
    "azido": ("azido", "click partner"),
    "propargyl": ("prop-2-yn-1-yl", "click partner"),
    "ethynyl": ("ethynyl", "click partner"),
    "vinyl": ("ethenyl", "polymerisable / cross-linkable"),
    "allyl": ("prop-2-en-1-yl", "polymerisable / cross-linkable"),
    "methallyl": ("2-methylprop-2-en-1-yl", "β-hydride elimination chain end"),
    "sulfanyl": ("sulfanyl", "thiol, for coupling or transfer"),
    "methylsulfanyl": ("methylsulfanyl", _ORIGIN_FUNCTIONAL),
    "maleimido": ("2,5-dioxo-2,5-dihydro-1H-pyrrol-1-yl", "thiol-coupling (bioconjugation)"),
    "succinimido": ("2,5-dioxopyrrolidin-1-yl", "activated ester (bioconjugation)"),
    "tosyloxy": ("4-methylbenzenesulfonyloxy", "activated leaving group"),
    "mesyloxy": ("methanesulfonyloxy", "activated leaving group"),
    "trimethylsilyl": ("trimethylsilyl", "protecting group"),
    "triethoxysilyl": ("triethoxysilyl", "silane coupling agent"),
    # --- ionic and inorganic ends ---
    "sulfo": ("sulfo", _ORIGIN_IONIC),
    "phosphono": ("phosphono", _ORIGIN_IONIC),
    "cyano": ("cyano", _ORIGIN_FUNCTIONAL),
    "nitro": ("nitro", _ORIGIN_FUNCTIONAL),
    "fluoro": ("fluoro", _ORIGIN_FUNCTIONAL),
    "iodo": ("iodo", _ORIGIN_FUNCTIONAL),
}

# ---------------------------------------------------------------------------
# Mechanism inference.
#
# The commodity table only covers ~40 polymers, so any monomer outside it still
# needs a mechanism to drive the end-group suggestions and the ordering hint.
# These are deliberately coarse: they answer "how do these monomers join", not
# "what catalyst would you use".
# ---------------------------------------------------------------------------
_VINYL = Chem.MolFromSmarts("[CX3]=[CX3]")
_STRAINED_RING = Chem.MolFromSmarts("[$([O,N,S;R1]);!$([n])]")
_CONDENSATION_HANDLES = [
    ("hydroxy", Chem.MolFromSmarts("[OX2H][#6]")),
    ("carboxylic acid", Chem.MolFromSmarts("[CX3](=O)[OX2H1]")),
    ("amine", Chem.MolFromSmarts("[NX3;H1,H2][#6]")),
    ("acyl halide", Chem.MolFromSmarts("[CX3](=O)[F,Cl,Br,I]")),
    ("isocyanate", Chem.MolFromSmarts("[NX2]=[CX2]=[OX1]")),
    ("ester", Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")),
]
#: An aryl halide pair is a cross-coupling monomer (P3HT and other conjugated
#: polymers), not a condensation handle in the classical sense.
_ARYL_HALIDE = Chem.MolFromSmarts("[c][F,Cl,Br,I]")

# End groups worth offering, per mechanism: initiator residues, transfer agents,
# or the unreacted handle left on the last monomer. Ordered most-likely first,
# because the page shows them in this order.
_END_GROUP_SUGGESTIONS = {
    "chain": [
        "hydro", "cyanoisopropyl (AIBN)", "benzoyloxy", "sulfooxy", "tert-butoxy",
        "bromo", "chloro", "TEMPO", "RAFT dithiobenzoate", "RAFT trithiocarbonate",
        "butyl", "sec-butyl", "phenyl", "hydroxy", "carboxy", "sulfanyl",
    ],
    "ring": [
        "hydroxy", "methoxy", "ethoxy", "hydro", "acetoxy", "acetyl", "amino",
        "butyl", "carboxy", "tosyloxy", "mesyloxy", "azido", "propargyl",
    ],
    "condensation": [
        "hydroxy", "carboxy", "amino", "isocyanato", "chlorocarbonyl", "formyl",
        "acetyl", "acetoxy", "methoxy", "chloro",
    ],
    "coupling": [
        "bromo", "chloro", "iodo", "hydro", "phenyl", "methyl",
    ],
}

# A reactive handle seen in the monomers is, by definition, a group that can be
# left unreacted on a chain end, so detecting one has to also offer it.
_HANDLE_END_GROUPS = {
    "hydroxy": "hydroxy",
    "carboxylic acid": "carboxy",
    "amine": "amino",
    "acyl halide": "chlorocarbonyl",
    "isocyanate": "isocyanato",
}


def _suggested_end_groups(mechanism: str, handles: list) -> list:
    """End groups plausible for this polymer, most likely first."""
    suggestions = list(_END_GROUP_SUGGESTIONS.get(mechanism, []))
    # Anything the monomers can actually leave dangling belongs near the front.
    for handle in handles:
        name = _HANDLE_END_GROUPS.get(handle)
        if name and name in _END_GROUPS:
            if name in suggestions:
                suggestions.remove(name)
            suggestions.insert(0, name)
    if not suggestions:
        suggestions = sorted(_END_GROUPS)
    return [name for name in suggestions if name in _END_GROUPS]

# How the monomer order in the builder should be read, per mechanism.
_ORDER_HINTS = {
    "chain": "Addition polymerisation: the monomers join through their double bonds and the backbone follows this order.",
    "ring": "Ring-opening polymerisation: each ring opens and adds to the chain end.",
    "condensation": "Step-growth: these monomers condense and expel a small molecule, so they alternate along the chain regardless of the order shown.",
    "coupling": "Metal-catalysed cross-coupling: the aryl halide ends couple into a conjugated backbone (Kumada/GRIM, Suzuki, Stille).",
    "unknown": "Mechanism not recognised, so the order shown is taken at face value.",
}


def _polymer_canon(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# Commodity polymer table.
#
# Keyed by the canonical SMILES of the monomer(s), so a drawn structure matches
# regardless of how it was written. Each entry carries the trivial name chemists
# actually use, its acronym, the polymerisation class, and -- where a standard
# one exists -- the structure-based CRU name.
# ---------------------------------------------------------------------------
_COMMODITY = [
    # --- chain-growth (vinyl / olefin addition) ---
    (["C=C"], "polyethylene", "PE", "chain", "poly(methylene)"),
    (["C=CC"], "polypropylene", "PP", "chain", "poly(propane-1,2-diyl)"),
    (["C=CC=C"], "polybutadiene", "PB", "chain", None),
    (["C=CC(=C)C"], "polyisoprene", "PI", "chain", None),
    (["C=Cc1ccccc1"], "polystyrene", "PS", "chain", "poly(1-phenylethane-1,2-diyl)"),
    (["C=CCl"], "poly(vinyl chloride)", "PVC", "chain", "poly(1-chloroethane-1,2-diyl)"),
    (["C=C(Cl)Cl"], "poly(vinylidene chloride)", "PVDC", "chain", None),
    (["C=CF"], "poly(vinyl fluoride)", "PVF", "chain", None),
    (["C=C(F)F"], "poly(vinylidene fluoride)", "PVDF", "chain", None),
    (["FC(F)=C(F)F"], "polytetrafluoroethylene", "PTFE", "chain", "poly(difluoromethylene)"),
    (["C=CC#N"], "polyacrylonitrile", "PAN", "chain", None),
    (["C=CC(=O)O"], "poly(acrylic acid)", "PAA", "chain", None),
    (["C=CC(N)=O"], "polyacrylamide", "PAM", "chain", None),
    (["C=CC(=O)OC"], "poly(methyl acrylate)", "PMA", "chain", None),
    (["C=C(C)C(=O)OC"], "poly(methyl methacrylate)", "PMMA", "chain", None),
    (["C=C(C)C(=O)O"], "poly(methacrylic acid)", "PMAA", "chain", None),
    (["C=COC(C)=O"], "poly(vinyl acetate)", "PVAc", "chain", None),
    (["C=CO"], "poly(vinyl alcohol)", "PVA", "chain", None),
    (["C=CN1CCCC1=O"], "poly(vinylpyrrolidone)", "PVP", "chain", None),
    (["C=CCC(C)C"], "poly(4-methylpent-1-ene)", "PMP", "chain", None),
    # --- ring-opening ---
    (["C1CO1"], "poly(ethylene oxide)", "PEO/PEG", "ring", "poly(oxyethane-1,2-diyl)"),
    (["CC1CO1"], "poly(propylene oxide)", "PPO/PPG", "ring", "poly[oxy(1-methylethane-1,2-diyl)]"),
    (["C1CCCO1"], "poly(tetrahydrofuran)", "PTHF", "ring", "poly(oxybutane-1,4-diyl)"),
    (["O=C1CCCCCO1"], "poly(caprolactone)", "PCL", "ring", None),
    (["O=C1CCCCCN1"], "polyamide 6", "PA6", "ring", None),
    (["CC1OC(=O)C(C)OC1=O"], "poly(lactic acid)", "PLA", "ring", None),
    (["C1COC(=O)O1"], "poly(ethylene carbonate)", "PEC", "ring", None),
    (["C1CN1"], "poly(ethyleneimine)", "PEI", "ring", "poly(iminoethane-1,2-diyl)"),
    # --- step-growth (condensation pairs) ---
    (["OCCO", "OC(=O)c1ccc(C(=O)O)cc1"], "poly(ethylene terephthalate)", "PET", "condensation", None),
    (["OCCCCO", "OC(=O)c1ccc(C(=O)O)cc1"], "poly(butylene terephthalate)", "PBT", "condensation", None),
    (["NCCCCCCN", "OC(=O)CCCCC(=O)O"], "polyamide 6,6", "PA66", "condensation", None),
    (["NCCCCCCN", "OC(=O)CCCCCCCCC(=O)O"], "polyamide 6,10", "PA610", "condensation", None),
    (["Nc1ccc(N)cc1", "OC(=O)c1ccc(C(=O)O)cc1"], "poly(p-phenylene terephthalamide)", "PPTA", "condensation", None),
    (["CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", "O=C(Cl)Cl"], "polycarbonate (bisphenol A)", "PC", "condensation", None),
    (["OCCO", "OC(=O)CCCCC(=O)O"], "poly(ethylene adipate)", "PEA", "condensation", None),
    # --- common copolymers, matched as monomer sets ---
    (["C=Cc1ccccc1", "C=CC=C"], "poly(styrene-co-butadiene)", "SBR", "chain", None),
    (["C=Cc1ccccc1", "C=CC#N"], "poly(styrene-co-acrylonitrile)", "SAN", "chain", None),
    (["C=C", "C=CC"], "poly(ethylene-co-propylene)", "EPR", "chain", None),
    (["C=C", "C=COC(C)=O"], "poly(ethylene-co-vinyl acetate)", "EVA", "chain", None),
]


def _build_commodity_index() -> dict:
    """Canonicalise the table at import so lookups match any input spelling."""
    index = {}
    for smiles_list, trivial, acronym, mechanism, cru in _COMMODITY:
        keys = []
        for smi in smiles_list:
            canon = _polymer_canon(smi)
            if canon is None:
                raise ValueError(f"bad commodity SMILES: {smi}")
            keys.append(canon)
        index[frozenset(keys)] = {
            "trivial_name": trivial,
            "acronym": acronym,
            "mechanism": mechanism,
            "cru_name": cru,
            "monomer_count": len(keys),
        }
    return index


_COMMODITY_INDEX = _build_commodity_index()

# ---------------------------------------------------------------------------
# Trivial monomer names.
#
# Source-based names are written with the monomer name chemists use, so a
# copolymer reads poly(ethylene-co-styrene) rather than
# poly[(eth-1-ene)-co-ethenylbenzene]. The commodity table above cannot supply
# this: it is keyed on the whole polymer, so it only fires for the ~40 curated
# ones and never for an arbitrary copolymer. Naming the monomers instead
# composes -- any combination of the monomers below gets a trivial name.
#
# These are deliberately the polymer-chemistry names ("vinyl chloride",
# "ethylene"), which are often not the current IUPAC preferred names. That is the
# point: the systematic name is already the page's primary answer.
# ---------------------------------------------------------------------------
_MONOMER_TRIVIAL = {
    # olefins and dienes
    "C=C": "ethylene",
    "C=CC": "propylene",
    "C=CCC": "but-1-ene",
    "C=CCC(C)C": "4-methylpent-1-ene",
    "C=CC=C": "butadiene",
    "C=CC(=C)C": "isoprene",
    "CC(=C)C": "isobutylene",
    # styrenics
    "C=Cc1ccccc1": "styrene",
    "C=Cc1ccc(C)cc1": "4-methylstyrene",
    # vinyl halides and fluoro monomers
    "C=CCl": "vinyl chloride",
    "C=C(Cl)Cl": "vinylidene chloride",
    "C=CF": "vinyl fluoride",
    "C=C(F)F": "vinylidene fluoride",
    "FC(F)=C(F)F": "tetrafluoroethylene",
    # acrylics
    "C=CC#N": "acrylonitrile",
    "C=CC(=O)O": "acrylic acid",
    "C=CC(N)=O": "acrylamide",
    "C=CC(=O)OC": "methyl acrylate",
    "C=CC(=O)OCC": "ethyl acrylate",
    "C=CC(=O)OCCCC": "butyl acrylate",
    "C=C(C)C(=O)O": "methacrylic acid",
    "C=C(C)C(=O)OC": "methyl methacrylate",
    # vinyl esters, ethers and amides
    "C=COC(C)=O": "vinyl acetate",
    "C=CO": "vinyl alcohol",
    "C=CN1CCCC1=O": "N-vinylpyrrolidone",
    # cyclic monomers
    "C1CO1": "ethylene oxide",
    "CC1CO1": "propylene oxide",
    "C1CCCO1": "tetrahydrofuran",
    "C1CN1": "ethyleneimine",
    "O=C1CCCCCO1": "ε-caprolactone",
    "O=C1CCCCCN1": "ε-caprolactam",
    "CC1OC(=O)C(C)OC1=O": "lactide",
    "C1COC(=O)O1": "ethylene carbonate",
    # step-growth diols, diacids, diamines and friends
    "OCCO": "ethylene glycol",
    "OCCCCO": "butane-1,4-diol",
    "OCCCCCCO": "hexane-1,6-diol",
    "OCCOCCO": "diethylene glycol",
    "OC(=O)c1ccc(C(=O)O)cc1": "terephthalic acid",
    "OC(=O)c1cccc(C(=O)O)c1": "isophthalic acid",
    "OC(=O)CCCCC(=O)O": "adipic acid",
    "OC(=O)CCCCCCCCC(=O)O": "sebacic acid",
    "NCCCCCCN": "hexamethylenediamine",
    "NCCN": "ethylenediamine",
    "Nc1ccc(N)cc1": "p-phenylenediamine",
    "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1": "bisphenol A",
    "O=C(Cl)Cl": "phosgene",
    "O=C=Nc1ccc(Cc2ccc(N=C=O)cc2)cc1": "methylenediphenyl diisocyanate",
    "O=C=Nc1ccc(N=C=O)cc1": "1,4-phenylene diisocyanate",
    "O=C=NCCCCCCN=C=O": "hexamethylene diisocyanate",
    "Cc1ccc(N=C=O)cc1N=C=O": "toluene diisocyanate",
}


def _build_monomer_trivial_index() -> dict:
    index = {}
    for smi, trivial in _MONOMER_TRIVIAL.items():
        canon = _polymer_canon(smi)
        if canon is None:
            raise ValueError(f"bad monomer SMILES: {smi}")
        index[canon] = trivial
    return index


_MONOMER_TRIVIAL_INDEX = _build_monomer_trivial_index()

# IUPAC nests enclosing marks () -> [] -> {}; a name that already contains a
# mark has to be wrapped in the next one out so the nesting stays readable.
_MARKS = [("(", ")"), ("[", "]"), ("{", "}")]


def _mark_depth(text: str) -> int:
    """Return the outermost mark level already used inside ``text``."""
    depth = 0
    for level, (open_mark, _) in enumerate(_MARKS, start=1):
        if open_mark in text:
            depth = level
    return depth


def _enclose(text: str) -> str:
    level = min(_mark_depth(text), len(_MARKS) - 1)
    open_mark, close_mark = _MARKS[level]
    return f"{open_mark}{text}{close_mark}"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _enclose_prefix(prefix: str) -> str:
    """Enclose a composite end-group prefix so it reads as one unit.

    A simple prefix (methoxy, bromo, sec-butyl) needs nothing. One carrying its
    own locants runs into the neighbouring alpha/omega hyphens and becomes
    ambiguous -- "ω-2,5-dioxo...pyrrol-1-ylpoly(oxirane)" -- so anything with a
    digit, a space or its own enclosing mark gets wrapped.
    """
    if any(ch.isdigit() or ch == " " or ch in "([{" for ch in prefix):
        return _enclose(prefix)
    return prefix


def _all_aromatic_alkene(mol) -> bool:
    """True when every C=C match is part of an aromatic ring (not polymerisable)."""
    matches = mol.GetSubstructMatches(_VINYL)
    if not matches:
        return True
    for a, b in matches:
        if not (mol.GetAtomWithIdx(a).GetIsAromatic() and mol.GetAtomWithIdx(b).GetIsAromatic()):
            return False
    return True


def _ring_opening_candidate(mol) -> bool:
    """True if the molecule has a small heteroatom-containing ring."""
    if not mol.HasSubstructMatch(_STRAINED_RING):
        return False
    for ring in mol.GetRingInfo().AtomRings():
        if not 3 <= len(ring) <= 7:
            continue
        # Aromatic rings do not open under polymerisation conditions.
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if any(mol.GetAtomWithIdx(i).GetSymbol() in ("O", "N", "S") for i in ring):
            return True
    return False


def _condensation_handles(mols: list) -> list[str]:
    """Names of the reactive handles present across the monomer set."""
    found = []
    for label, patt in _CONDENSATION_HANDLES:
        if patt is None:
            continue
        if any(m.GetSubstructMatches(patt) for m in mols):
            found.append(label)
    return found


def _infer_mechanism(mols: list) -> str:
    """Classify how a monomer set joins up: chain, ring or condensation."""
    # A vinyl group anywhere means addition polymerisation, which outranks a
    # ring in the same molecule (N-vinylpyrrolidone polymerises through the
    # double bond, not by opening the lactam).
    if any(m.HasSubstructMatch(_VINYL) and not _all_aromatic_alkene(m) for m in mols):
        return "chain"
    if any(_ring_opening_candidate(m) for m in mols):
        return "ring"
    if _condensation_handles(mols):
        return "condensation"
    # Two aryl halides on one monomer: a cross-coupling feed (dibromothiophene
    # and friends), which no condensation-handle pattern covers.
    if any(len(m.GetSubstructMatches(_ARYL_HALIDE)) >= 2 for m in mols):
        return "coupling"
    return "unknown"


def _end_group_prefix(choice: str | None) -> str:
    """Resolve an end-group choice to its IUPAC substituent prefix.

    A choice that is not in the table is passed through as typed, so a caller can
    supply a prefix the curated set does not cover.
    """
    key = (choice or "").strip()
    if not key:
        return ""
    entry = _END_GROUPS.get(key)
    return entry[0] if entry else key


def _needs_marks(name: str) -> bool:
    """Whether a monomer name needs enclosing marks after "poly".

    The Brief Guide to Polymer Nomenclature writes polystyrene and
    polypropylene but poly(vinyl chloride) and poly(prop-1-ene): marks are
    required when the name carries locants, spaces or its own enclosures, and
    omitted for a single simple word.
    """
    return any(ch.isdigit() or ch in " -([{" for ch in name)


def _poly(name: str, marks: str) -> str:
    if marks == "as-needed" and not _needs_marks(name):
        return "poly" + name
    return "poly" + _enclose(name)


def _assemble_name(
    monomer_names: list,
    connective: str,
    architecture: str,
    end_prefix: str,
    marks: str = "always",
    tacticity: str | None = None,
) -> tuple[str, str, str]:
    """Wrap monomer names into a source-based polymer name.

    Returns (name, name_html, core). Run over the systematic monomer names for
    the primary name, and over their trivial names for the cross-reference.

    marks: "always" encloses every monomer name -- the safe choice for
    systematic names, where "polyethenylbenzene" would invite the wrong parse.
    "as-needed" follows the Brief Guide style used with trivial monomer names:
    polystyrene, polybutadiene, but poly(vinyl chloride).
    """
    _, joins_blocks = _CONNECTIVES[connective]
    multi = len(monomer_names) > 1

    # A connective is only meaningful between two or more monomers; on a
    # homopolymer it is dropped rather than reported as part of the name.
    if not multi:
        core = _poly(monomer_names[0], marks)
        core_html = _esc(core)
    elif joins_blocks:
        # poly(A)-block-poly(B): each block keeps its own poly() wrapper.
        blocks = [_poly(n, marks) for n in monomer_names]
        core = f"-{connective}-".join(blocks)
        core_html = f"-<i>{connective}</i>-".join(_esc(b) for b in blocks)
    else:
        # poly(A-co-B): one wrapper around the joined monomer names. A monomer
        # name that already contains a hyphen or space is enclosed on its own,
        # otherwise the connective disappears among the locant hyphens.
        parts = [_enclose(n) if (("-" in n) or (" " in n)) else n for n in monomer_names]
        joined = f"-{connective}-".join(parts)
        core = "poly" + _enclose(joined)
        level = min(_mark_depth(joined), len(_MARKS) - 1)
        open_mark, close_mark = _MARKS[level]
        inner = f"-<i>{connective}</i>-".join(_esc(p) for p in parts)
        core_html = f"poly{open_mark}{inner}{close_mark}"

    arch = _ARCHITECTURES[architecture]
    # Qualifier order: architecture, then configuration, then the α/ω end
    # groups running straight into "poly": cyclo-it-α-methoxy-...poly(...).
    tact = f"{tacticity}-" if tacticity else ""
    name = f"{arch + '-' if arch else ''}{tact}{end_prefix}{core}"
    name_html = (f"{'<i>' + arch + '</i>-' if arch else ''}"
                 f"{'<i>' + tacticity + '</i>-' if tacticity else ''}"
                 f"{_esc(end_prefix)}{core_html}")
    return name, name_html, core


def _monomer_name(smiles: str, override: str | None) -> tuple[str | None, str | None]:
    """Name one monomer with the ordinary backbone. Returns (name, error)."""
    if override and override.strip():
        return override.strip(), None
    if _polymer_canon(smiles) is None:
        return None, f"Not a valid structure: {smiles}"
    result = name_one(smiles)
    if result.error or not result.name:
        return None, "This monomer could not be named yet."
    return result.name, None


def _end_prefix_from(end_alpha: str | None, end_omega: str | None) -> str:
    """The α/ω prefix run, empty when no end groups are chosen."""
    alpha = _end_group_prefix(end_alpha)
    omega = _end_group_prefix(end_omega)
    end_parts = []
    if alpha:
        end_parts.append(f"α-{_enclose_prefix(alpha)}")
    if omega:
        end_parts.append(f"ω-{_enclose_prefix(omega)}")
    return "-".join(end_parts) if end_parts else ""


def _named_monomer(entry) -> tuple[dict | None, dict | None]:
    """Resolve one monomer entry to its named card, or (None, error payload)."""
    if isinstance(entry, str):
        entry = {"smiles": entry}
    smiles = (entry.get("smiles") or "").strip()
    if not smiles:
        return None, {"ok": False, "error": "A monomer has no structure."}
    mono_name, error = _monomer_name(smiles, entry.get("name"))
    if error:
        return None, {"ok": False, "error": error, "smiles": smiles}
    canonical = _polymer_canon(smiles)
    return {
        "smiles": smiles,
        "canonical": canonical,
        "name": mono_name,
        "trivial_name": _MONOMER_TRIVIAL_INDEX.get(canonical),
    }, None


def _build_block_tree_name(
    blocks: list,
    architecture: str,
    end_alpha: str | None,
    end_omega: str | None,
    degree: int | None,
    tacticity: str | None,
) -> dict:
    """Name a polymer given per-block structure, not a flat monomer list.

    Each block keeps its own inner connective, so
    poly(A)-block-poly(B-stat-C)-block-poly(A) is representable; consecutive
    blocks are joined by each block's relation to its predecessor (block,
    graft or blend). This is the structural information a flat monomer list
    cannot hold, which is why reading a triblock used to degrade.
    """
    normalized, named_flat = [], []
    for blk in blocks:
        entries = blk.get("monomers") or []
        if not entries:
            return {"ok": False, "error": "A block has no monomers."}
        block_named = []
        for entry in entries:
            item, error = _named_monomer(entry)
            if error:
                return error
            block_named.append(item)
            named_flat.append(item)
        inner = blk.get("connective") or "co"
        if inner not in _CONNECTIVES or _CONNECTIVES[inner][1]:
            # A block's internal connective joins monomers, never blocks.
            inner = "co"
        relation = blk.get("relation") or "block"
        if relation not in ("block", "graft", "blend"):
            relation = "block"
        normalized.append(
            {"named": block_named, "connective": inner, "relation": relation})

    end_prefix = _end_prefix_from(end_alpha, end_omega)

    def assemble(marks: str, key: str) -> tuple[str, str] | None:
        cores, cores_html = [], []
        for blk in normalized:
            names = [item[key] for item in blk["named"]]
            if not all(names):
                return None
            sub, sub_html, _ = _assemble_name(
                names, blk["connective"], "linear", "", marks=marks)
            cores.append(sub)
            cores_html.append(sub_html)
        parts, parts_html = [cores[0]], [cores_html[0]]
        for position in range(1, len(cores)):
            relation = normalized[position]["relation"]
            parts.append(f"-{relation}-{cores[position]}")
            parts_html.append(f"-<i>{relation}</i>-{cores_html[position]}")
        return "".join(parts), "".join(parts_html)

    core, core_html = assemble("always", "name")
    arch = _ARCHITECTURES[architecture]
    tact = f"{tacticity}-" if tacticity else ""
    name = f"{arch + '-' if arch else ''}{tact}{end_prefix}{core}"
    name_html = (f"{'<i>' + arch + '</i>-' if arch else ''}"
                 f"{'<i>' + tacticity + '</i>-' if tacticity else ''}"
                 f"{_esc(end_prefix)}{core_html}")

    trivial = assemble("as-needed", "trivial_name")
    trivial_name = None
    if trivial:
        trivial_name = f"{arch + '-' if arch else ''}{tact}{end_prefix}{trivial[0]}"

    mols = [m for m in (Chem.MolFromSmiles(item["smiles"]) for item in named_flat)
            if m is not None]
    mechanism = _infer_mechanism(mols)
    handles = _condensation_handles(mols)
    relations = {blk["relation"] for blk in normalized[1:]} or {"block"}
    connective = sorted(relations)[0] if len(relations) == 1 else "block"
    return {
        "ok": True,
        "name": name,
        "name_html": name_html,
        "core": core,
        "trivial_name": trivial_name,
        "monomers": named_flat,
        "blocks": [{"monomers": [i["smiles"] for i in blk["named"]],
                    "connective": blk["connective"],
                    "relation": blk["relation"]} for blk in normalized],
        "connective": connective,
        "connective_meaning": _CONNECTIVES[connective][0],
        "architecture": architecture,
        "tacticity": tacticity,
        "end_alpha": _end_group_prefix(end_alpha) or None,
        "end_omega": _end_group_prefix(end_omega) or None,
        "degree": degree,
        "commodity": None,
        "mechanism": mechanism,
        "order_hint": _ORDER_HINTS.get(mechanism),
        "handles": handles,
        "suggested_end_groups": _suggested_end_groups(mechanism, handles),
    }


def build_polymer_name(
    monomers: list,
    connective: str = "co",
    architecture: str = "linear",
    end_alpha: str | None = None,
    end_omega: str | None = None,
    degree: int | None = None,
    tacticity: str | None = None,
    blocks: list | None = None,
) -> dict:
    """Assemble a source-based polymer name from its monomers.

    ``blocks`` overrides the flat monomer list with per-block structure -- each
    block its own monomers and inner connective -- for polymers a flat list
    cannot describe (a triblock whose middle block is itself a copolymer).
    """
    tacticity = (tacticity or "").strip() or None
    if tacticity and tacticity not in _TACTICITIES:
        return {"ok": False, "error": f"Unknown tacticity: {tacticity}"}
    if architecture not in _ARCHITECTURES:
        return {"ok": False, "error": f"Unknown architecture: {architecture}"}
    if blocks:
        return _build_block_tree_name(
            blocks, architecture, end_alpha, end_omega, degree, tacticity)
    if not monomers:
        return {"ok": False, "error": "Add at least one monomer."}
    if connective not in _CONNECTIVES:
        return {"ok": False, "error": f"Unknown connective: {connective}"}

    named = []
    for entry in monomers:
        item, error = _named_monomer(entry)
        if error:
            return error
        named.append(item)

    mols = [m for m in (Chem.MolFromSmiles(item["smiles"]) for item in named) if m is not None]

    # Commodity lookup, on the set of monomers actually present.
    key = frozenset(item["canonical"] for item in named)
    commodity = _COMMODITY_INDEX.get(key)
    # A set-keyed match must agree on how many distinct monomers there are, so a
    # homopolymer never matches a copolymer entry that happens to share a monomer.
    if commodity and commodity["monomer_count"] != len(key):
        commodity = None
    if commodity:
        commodity = dict(commodity)
        # The trivial name of a copolymer assumes a particular monomer
        # arrangement: SBR is statistical, PET strictly alternating. Sharing the
        # monomers is not the same polymer if the requested arrangement differs,
        # so say "related" rather than asserting the trivial name.
        expected = None
        if commodity["monomer_count"] > 1:
            expected = "alt" if commodity["mechanism"] == "condensation" else "ran"
        commodity["expected_connective"] = expected
        commodity["match"] = "exact" if expected is None or connective in ("co", expected) else "related"

    monomer_names = [item["name"] for item in named]
    multi = len(monomer_names) > 1

    # End groups sit in front as alpha/omega substituent prefixes. The last one
    # runs straight into "poly" with no hyphen.
    alpha = _end_group_prefix(end_alpha)
    omega = _end_group_prefix(end_omega)
    end_prefix = _end_prefix_from(end_alpha, end_omega)

    name, name_html, core = _assemble_name(
        monomer_names, connective, architecture, end_prefix, tacticity=tacticity)

    # The same assembly over the monomers' trivial names, which is how chemists
    # actually write these: poly(ethylene-co-styrene), not
    # poly[(eth-1-ene)-co-ethenylbenzene]. Only offered when every monomer has a
    # trivial name, since a half-trivial name belongs to no convention at all.
    trivial_parts = [_MONOMER_TRIVIAL_INDEX.get(item["canonical"]) for item in named]
    trivial_name = None
    if all(trivial_parts):
        trivial_name, _, _ = _assemble_name(
            trivial_parts, connective, architecture, end_prefix,
            marks="as-needed", tacticity=tacticity)

    mechanism = (commodity or {}).get("mechanism") or _infer_mechanism(mols)
    handles = _condensation_handles(mols)
    return {
        "ok": True,
        "name": name,
        "name_html": name_html,
        "core": core,
        # The same polymer written with trivial monomer names. Present for any
        # combination of known monomers, so copolymers get one too.
        "trivial_name": trivial_name,
        "monomers": named,
        "connective": connective if multi else None,
        "connective_meaning": _CONNECTIVES[connective][0] if multi else None,
        "architecture": architecture,
        "tacticity": tacticity,
        "end_alpha": alpha or None,
        "end_omega": omega or None,
        "degree": degree,
        "commodity": commodity,
        "mechanism": mechanism,
        "order_hint": _ORDER_HINTS.get(mechanism),
        "handles": handles,
        "suggested_end_groups": _suggested_end_groups(mechanism, handles),
    }


class PolymerMonomer(BaseModel):
    smiles: str
    name: str | None = None


class PolymerBlock(BaseModel):
    monomers: list[PolymerMonomer] = Field(default_factory=list, max_length=8)
    connective: str = "co"          # within the block: co/stat/ran/alt/grad/per
    relation: str = "block"         # to the previous block: block/graft/blend


class PolymerRequest(BaseModel):
    monomers: list[PolymerMonomer] = Field(default_factory=list, max_length=8)
    connective: str = "co"
    architecture: str = "linear"
    end_alpha: str | None = None
    end_omega: str | None = None
    degree: int | None = Field(None, ge=1)
    tacticity: str | None = None
    # Per-block structure for polymers a flat monomer list cannot describe
    # (a triblock whose middle block is itself a copolymer). When present it
    # overrides ``monomers`` and ``connective``.
    blocks: list[PolymerBlock] | None = Field(None, max_length=8)
    verify: bool = True
    no_cache: bool = False


@app.post("/api/polymer")
def polymer_endpoint(req: PolymerRequest) -> dict:
    """Build a source-based polymer name, for openclatura.org/polymers.

    ``verify`` round-trips each *monomer* name through OPSIN. There is nothing
    to verify about the poly(...) assembly itself -- OPSIN cannot read
    source-based polymer names -- but the monomer names are where all the real
    naming happens, so checking them checks the part that can be wrong.
    """
    blocks = ([b.model_dump() for b in req.blocks] if req.blocks else None)
    flags = "|".join(
        [
            req.connective,
            req.architecture,
            req.end_alpha or "",
            req.end_omega or "",
            str(req.degree or ""),
            req.tacticity or "",
            repr(blocks) if blocks else "",
            str(int(req.verify)),
            ";".join(f"{m.smiles}>{m.name or ''}" for m in req.monomers),
        ]
    )
    key = _cache_key("polymer", flags, "v2")
    cached = _cache_get(key)
    if cached is not None:
        return cached

    payload = build_polymer_name(
        [m.model_dump() for m in req.monomers],
        connective=req.connective,
        architecture=req.architecture,
        end_alpha=req.end_alpha,
        end_omega=req.end_omega,
        degree=req.degree,
        tacticity=req.tacticity,
        blocks=blocks,
    )
    if not payload.get("ok"):
        return payload

    if req.verify:
        with _OPSIN_LOCK:
            for item in payload["monomers"]:
                check = _opsin_check_via_daemon(item["name"], item["smiles"])
                item["opsin"] = check.status if check is not None else None

    # The same polymer in BigSMILES, so the name and the machine-readable
    # structure are shown together and can be checked against each other,
    # chosen end groups included. An end group with no fragment in the curated
    # map is left out of the string rather than guessed.
    payload["bigsmiles"] = write_bigsmiles(
        [{"smiles": item["smiles"]} for item in payload["monomers"]],
        connective=req.connective,
        end_alpha=_end_group_to_smiles(req.end_alpha),
        end_omega=_end_group_to_smiles(req.end_omega),
        blocks=payload.get("blocks") if blocks else None,
        tacticity=req.tacticity,
    )

    if not req.no_cache:
        _cache_set(key, payload)
    return payload


# Imported end groups arrive as SMILES fragments with a [*] attachment point;
# the builder holds end groups as curated names. This maps one onto the other,
# keyed on canonical fragment SMILES so any input spelling matches. A fragment
# with no entry is reported but left out of the name -- an honest gap beats a
# guessed prefix.
_END_GROUP_BY_SMILES = {}


def _register_end_group_smiles() -> None:
    pairs = [
        ("[*][H]", "hydro"),
        ("[*]C", "methyl"),
        ("[*]CC", "ethyl"),
        ("[*]CCC", "propyl"),
        ("[*]CCCC", "butyl"),
        ("[*]C(C)(C)C", "tert-butyl"),
        ("[*]c1ccccc1", "phenyl"),
        ("[*]Cc1ccccc1", "benzyl"),
        ("[*]O", "hydroxy"),
        ("[*]OC", "methoxy"),
        ("[*]OCC", "ethoxy"),
        ("[*]C(=O)O", "carboxy"),
        ("[*]C=O", "formyl"),
        ("[*]C(C)=O", "acetyl"),
        ("[*]OC(C)=O", "acetoxy"),
        ("[*]N", "amino"),
        ("[*]N=C=O", "isocyanato"),
        ("[*]C(=O)Cl", "chlorocarbonyl"),
        ("[*]Cl", "chloro"),
        ("[*]Br", "bromo"),
        ("[*]F", "fluoro"),
        ("[*]I", "iodo"),
        ("[*]C=C", "vinyl"),
        ("[*]CC=C", "allyl"),
        ("[*]CC(C)=C", "methallyl"),
        ("[*]C#C", "ethynyl"),
        ("[*]CC#C", "propargyl"),
        ("[*]C#N", "cyano"),
        ("[*]S", "sulfanyl"),
        ("[*]SC", "methylsulfanyl"),
        ("[*]N=[N+]=[N-]", "azido"),
    ]
    for smi, key in pairs:
        try:
            _END_GROUP_BY_SMILES[Chem.CanonSmiles(smi)] = key
        except Exception:
            continue


_register_end_group_smiles()


def _end_group_from_smiles(fragment: str | None) -> str | None:
    """Curated end-group name for an attachment-marked fragment, or None."""
    if not fragment:
        return None
    try:
        return _END_GROUP_BY_SMILES.get(Chem.CanonSmiles(fragment))
    except Exception:
        return None


#: The reverse direction: curated end-group name -> [*]-marked fragment, for
#: putting the chosen ends into the emitted BigSMILES.
_END_GROUP_TO_SMILES = {key: smi for smi, key in _END_GROUP_BY_SMILES.items()}


def _end_group_to_smiles(choice: str | None) -> str | None:
    return _END_GROUP_TO_SMILES.get((choice or "").strip() or None)


class PolymerImportRequest(BaseModel):
    text: str = Field(..., max_length=4000)
    route_choice: dict[str, str] = Field(default_factory=dict)


@app.post("/api/polymer/import")
def polymer_import(req: PolymerImportRequest) -> dict:
    """Pasted text -> monomer cards, for openclatura.org/polymers.

    The page sends whatever was pasted and this decides how to read it. ``{`` and
    ``}`` are not in the SMILES grammar at any position, so a brace is a
    conclusive signal for BigSMILES; anything else is handed back as a plain
    structure for the ordinary single-monomer path.

    What comes back is builder state, not a name: monomers, the connective read
    off the bond descriptors, the end groups, and -- importantly -- the
    disconnection route that was assumed, with its alternatives. A route the
    polymer graph cannot settle (terephthalic acid vs dimethyl terephthalate for
    PET) is reported as an ambiguity so the page can offer the choice rather than
    bake a guess into the name.
    """
    text = (req.text or "").strip()
    if not text:
        return {"ok": False, "error": "Nothing to import."}

    if not looks_like_bigsmiles(text):
        return {"ok": True, "kind": "smiles", "monomers": [{"smiles": text}]}

    result = read_bigsmiles(text, req.route_choice or None)
    if not result.get("ok"):
        return result

    blocks = result["blocks"]

    # The ordered block sequence, graft side chains spliced in after their
    # backbone block and mixture components joined by "blend". This is the
    # per-block truth the reading direction produces; flattening it away is
    # what used to degrade poly(A)-block-poly(B-stat-C)-block-poly(A).
    sequence = []

    def _push(block, relation):
        sequence.append((block, relation))
        for graft in block.get("grafts") or []:
            # A nested object pendant to the unit is a graft; one spliced into
            # the backbone is a segment of the main chain, which is a block.
            _push(graft, "graft" if graft.get("attachment") != "backbone"
                  else "block")

    previous_component = None
    for block in blocks:
        relation = None
        if sequence:
            relation = ("blend" if block.get("component") != previous_component
                        else "block")
        _push(block, relation)
        previous_component = block.get("component")

    warnings = [w for block, _ in sequence for w in (block.get("warnings") or [])]
    tacticity = next((b.get("tacticity") for b, _ in sequence
                      if b.get("tacticity")), None)

    # Flatten to the builder's shape: one monomer list, one connective. A
    # backbone monomer carrying a graft site is capped with hydrogen for the
    # card -- an explicit assumption, since the site's substituent is the whole
    # side chain.
    monomers, seen = [], set()
    builder_blocks = []
    for block, relation in sequence:
        card_smiles, card_by_raw = [], {}
        # Whether this block's nested object sits in the backbone is a fact
        # about the repeat unit, so it is read once here rather than guessed
        # from each monomer, which no longer carries the chain ends.
        segmented = any(g.get("attachment") == "backbone"
                        for g in (block.get("grafts") or []))
        for item in block["monomers"]:
            entry = dict(item)
            raw = entry["smiles"]
            if "[1*]" in entry["smiles"]:
                if segmented:
                    # The nested object is spliced into the backbone, so this
                    # is a macromonomer -- an oligomer, not a small molecule.
                    # There is no honest monomer card for it; the nested object
                    # is what describes it, and capping one anyway turned a PEG
                    # chain extender into "ethanol".
                    warnings.append(
                        f"{entry['smiles']} is a macromonomer: the nested "
                        "object is spliced into its backbone, so it is named "
                        "as its own block rather than as a monomer")
                    entry["macromonomer"] = True
                    continue
                capped = strip_graft_sites(entry["smiles"])
                if capped:
                    warnings.append(
                        f"the backbone monomer {entry['smiles']} carries a "
                        "graft site; it is shown with that site "
                        "hydrogen-capped")
                    entry["graft_smiles"] = entry["smiles"]
                    entry["smiles"] = capped
                    entry["graft_site"] = True
            card_smiles.append(entry["smiles"])
            card_by_raw[raw] = entry["smiles"]
            if entry["smiles"] not in seen:
                seen.add(entry["smiles"])
                monomers.append(entry)
        inner = block["connective"]
        if inner not in ("co", "stat", "ran", "alt", "grad", "per"):
            inner = "co"
        evidence = block.get("evidence") or {}
        networks = evidence.get("unit_networks")
        unit_monomers = evidence.get("unit_monomers")
        if ((block.get("type") or "").startswith("blend")
                and networks and unit_monomers):
            # Disjoint bonding networks in one object are separate polymers:
            # one builder block per network, joined by "blend", so a rebuild
            # does not collapse them into a copolymer that was never written.
            for position, group in enumerate(networks):
                group_smiles = []
                for unit_index in group:
                    for raw in unit_monomers[unit_index]:
                        smiles = card_by_raw.get(raw, raw)
                        if smiles not in group_smiles:
                            group_smiles.append(smiles)
                if group_smiles:
                    builder_blocks.append({
                        "monomers": group_smiles,
                        "connective": "stat" if len(group) > 1 else "co",
                        "relation": ((relation or "block")
                                     if position == 0 else "blend"),
                    })
            continue
        if card_smiles:
            builder_blocks.append({
                "monomers": card_smiles,
                "connective": inner,
                "relation": relation or "block",
            })

    if len(builder_blocks) > 1:
        connective = builder_blocks[1]["relation"]
    elif blocks:
        connective = blocks[0]["connective"]
    else:
        connective = "co"
    if len(monomers) < 2:
        connective = "co"

    architecture = "linear"
    if result.get("mixture") or any(
            (b.get("type") or "").startswith("blend") for b, _ in sequence):
        architecture = "blend"

    # Outside-brace fragments belong to the whole chain: alpha comes from the
    # first block, omega from the last.
    alpha_smiles = blocks[0]["end_groups"].get("alpha") if blocks else None
    omega_smiles = blocks[-1]["end_groups"].get("omega") if blocks else None
    # End groups stated in the object's ";" list are just as real. Whether
    # their assignment to the two chain ends is settled depends on the bond
    # descriptors: two distinct directional descriptors ([>]O with [<][H]) can
    # each bond only one kind of site, so the pairing fixes which end is which.
    # Two identical descriptors ([$]CC with [$]CC(=C)C) fix nothing, and only
    # then is the written order an assumption worth flagging.
    ends_assumed_order = False
    if not alpha_smiles and not omega_smiles and blocks:
        stated = list(blocks[0]["end_groups"].get("stated") or [])
        descriptors = list(blocks[0]["end_groups"].get("stated_descriptors") or [])
        if stated:
            alpha_smiles = stated[0]
            omega_smiles = stated[1] if len(stated) > 1 else None
            distinct = (len(stated) == 2 and len(descriptors) == 2
                        and descriptors[0] != descriptors[1]
                        and "$" not in descriptors)
            ends_assumed_order = not distinct
    return {
        "ok": True,
        "kind": "bigsmiles",
        "bigsmiles": text,
        "monomers": monomers,
        "connective": connective,
        "end_alpha": alpha_smiles,
        "end_omega": omega_smiles,
        # The curated builder entry each fragment maps onto, when one exists;
        # this is what actually reaches the name.
        "end_alpha_choice": _end_group_from_smiles(alpha_smiles),
        "end_omega_choice": _end_group_from_smiles(omega_smiles),
        "ends_assumed_order": ends_assumed_order,
        "blocks": blocks,
        # Per-block builder state, so a multi-block import names as a tree
        # (poly(A)-block-poly(B-stat-C)-...) instead of one flat poly(...).
        "builder_blocks": builder_blocks if len(builder_blocks) > 1 else None,
        "architecture": architecture,
        "tacticity": tacticity,
        "warnings": warnings,
        "mixture": bool(result.get("mixture")),
        "n_blocks": result["n_blocks"],
        "ambiguities": [a for b, _ in sequence for a in b["ambiguities"]],
    }


@app.get("/api/polymer/options")
def polymer_options() -> dict:
    """Vocabulary for the builder's dropdowns, so the page has one source."""
    return {
        "ok": True,
        "connectives": [
            {"value": k, "meaning": v[0], "joins_blocks": v[1]} for k, v in _CONNECTIVES.items()
        ],
        "architectures": [{"value": k, "prefix": v} for k, v in _ARCHITECTURES.items()],
        "tacticities": [{"value": k, "meaning": v} for k, v in _TACTICITIES.items()],
        "end_groups": [
            {"value": name, "prefix": prefix, "origin": origin}
            for name, (prefix, origin) in sorted(_END_GROUPS.items())
        ],
        "commodity_count": len(_COMMODITY_INDEX),
    }


app.mount("/api", create_app())
