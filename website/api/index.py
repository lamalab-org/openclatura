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
import tarfile
import tempfile
import threading
import time
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
    no_cache: bool = False


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
    return _merge_unresolved(name, tokens)


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


app.mount("/api", create_app())
