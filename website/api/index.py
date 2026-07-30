"""Vercel serverless entrypoint.

Vercel passes the original request path (``/api/...``) to the ASGI app,
so the openclatura FastAPI app is mounted under ``/api``:

- POST /api/name      (supports ``verify_opsin`` for OPSIN round-trip)
- POST /api/batch
- POST /api/describe
- GET  /api/healthz   (also reports the release channel)

On the beta channel (``CHANNEL=beta``) every /api route except healthz sits
behind an email allowlist, served by:

- POST /api/beta/login   (mail a sign-in link)
- POST /api/beta/verify  (exchange the link for a session cookie)
- POST /api/beta/logout
- GET  /api/beta/status

The Vercel Python runtime ships no Java, which py2opsin needs for OPSIN
verification. A jlink-minimized JRE (java.base, java.xml, java.logging,
java.naming, java.management, java.desktop — the closure OPSIN's log4j
needs) is bundled as ``jre.tar.gz`` and extracted to ``/tmp`` on cold
start; its ``bin`` is prepended to ``PATH`` before openclatura checks
``shutil.which("java")``.
"""

import base64
import ctypes
import dataclasses
import hashlib
import importlib.metadata
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

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
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
# Release channel.
#
# The openclatura version is fixed at build time by requirements.txt, so one
# deployment serves exactly one version. Two Vercel projects share this code:
#
#   stable  CHANNEL unset (or "stable"), requirements.txt pins a PyPI release.
#   beta    CHANNEL=beta, requirements.txt installs from a git ref, and the
#           whole API is behind the email allowlist below.
#
# Each side advertises the other's URL via BETA_URL / STABLE_URL so the version
# chip in the header can offer the switch. BETA_GIT_REF is display-only: a git
# install reports the same version string as the release it branched from, so
# the ref is what actually tells you which build you are looking at.
# ---------------------------------------------------------------------------
_CHANNEL = (os.environ.get("CHANNEL") or "stable").strip().lower()
_IS_BETA = _CHANNEL == "beta"
_BETA_URL = (os.environ.get("BETA_URL") or "").rstrip("/")
_STABLE_URL = (os.environ.get("STABLE_URL") or "").rstrip("/")
_BETA_GIT_REF = os.environ.get("BETA_GIT_REF") or ""

# ---------------------------------------------------------------------------
# Beta access control: an email allowlist plus emailed magic links.
#
# BETA_ALLOWED_EMAILS is a comma/whitespace-separated list. A visitor enters an
# email; if it is on the list we mail a short-lived signed link, which the page
# exchanges for a signed 30-day session cookie. No shared password exists, so
# revoking access is just editing the env var (existing cookies die with the
# secret if you also rotate BETA_SESSION_SECRET).
#
# Required on the beta project: BETA_ALLOWED_EMAILS, BETA_SESSION_SECRET,
# RESEND_API_KEY, BETA_EMAIL_FROM. Without them beta stays locked, which is the
# safe direction to fail.
# ---------------------------------------------------------------------------
_BETA_SECRET = os.environ.get("BETA_SESSION_SECRET") or ""
_BETA_COOKIE = "openclatura_beta"
_BETA_LINK_TTL = 30 * 60
_BETA_SESSION_TTL = 30 * 24 * 3600
_RESEND_KEY = os.environ.get("RESEND_API_KEY") or ""
_BETA_EMAIL_FROM = os.environ.get("BETA_EMAIL_FROM") or ""


def _beta_allowlist() -> set[str]:
    raw = os.environ.get("BETA_ALLOWED_EMAILS") or ""
    return {part.strip().lower() for part in re.split(r"[,\s]+", raw) if part.strip()}


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64u_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _beta_sign(purpose: str, email: str, expires_at: int) -> str:
    """Return ``payload.signature``; the payload is readable but not forgeable."""
    import hmac

    payload = _b64u(f"{purpose}:{email}:{expires_at}".encode())
    sig = hmac.new(_BETA_SECRET.encode(), payload.encode(), hashlib.sha256).digest()
    return f"{payload}.{_b64u(sig)}"


def _beta_verify(purpose: str, token: str) -> str | None:
    """Return the email a valid, unexpired token was issued to, else None."""
    import hmac

    if not _BETA_SECRET or not token or "." not in token:
        return None
    payload, _, sig = token.rpartition(".")
    expected = hmac.new(_BETA_SECRET.encode(), payload.encode(), hashlib.sha256).digest()
    try:
        if not hmac.compare_digest(_b64u_decode(sig), expected):
            return None
        got_purpose, email, expires_at = _b64u_decode(payload).decode().split(":", 2)
    except Exception:
        return None
    if got_purpose != purpose or int(expires_at) < int(time.time()):
        return None
    # An allowlist edit revokes outstanding links and sessions immediately.
    return email if email in _beta_allowlist() else None


def _beta_session_email(request: Request) -> str | None:
    return _beta_verify("session", request.cookies.get(_BETA_COOKIE, ""))


def _send_magic_link(email: str, link: str) -> bool:
    """Mail the sign-in link via Resend. Returns False if sending failed."""
    if not (_RESEND_KEY and _BETA_EMAIL_FROM):
        return False
    body = json.dumps(
        {
            "from": _BETA_EMAIL_FROM,
            "to": [email],
            "subject": "Your openclatura beta sign-in link",
            "text": (
                "Open this link to unlock the openclatura beta site:\n\n"
                f"{link}\n\n"
                "The link is valid for 30 minutes and signs in this browser for 30 days.\n"
                "If you did not request it, you can ignore this email."
            ),
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=body,
        headers={"Authorization": f"Bearer {_RESEND_KEY}", "content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status < 300
    except Exception:
        return False


@app.middleware("http")
async def _beta_gate(request: Request, call_next):
    """On the beta deployment, require a session for every API route.

    Applied as middleware rather than per-endpoint so it also covers the
    openclatura app mounted at /api (e.g. /api/batch), which this module does
    not shadow. Static assets stay public: the UI reveals nothing, and the
    unlock screen has to be reachable.
    """
    path = request.url.path
    if not _IS_BETA or not path.startswith("/api/") or path.startswith("/api/beta/") or path == "/api/healthz":
        return await call_next(request)
    if _beta_session_email(request) is None:
        return JSONResponse({"ok": False, "error": "Beta access required."}, status_code=403)
    return await call_next(request)


class BetaLoginRequest(BaseModel):
    email: str
    # Where the emailed link should point; validated against this deployment's
    # own origin so the signed token can't be redirected to another host.
    origin: str = ""


@app.post("/api/beta/login")
def beta_login(req: BetaLoginRequest, request: Request) -> dict:
    """Email a magic link if the address is allowlisted.

    The response never reveals whether an address is on the list; it always
    reports that a link was sent if one could have been.
    """
    if not _IS_BETA:
        return {"ok": False, "error": "This deployment is not the beta channel."}
    if not (_BETA_SECRET and _RESEND_KEY and _BETA_EMAIL_FROM):
        return {"ok": False, "error": "Beta sign-in is not configured on this deployment."}
    email = req.email.strip().lower()
    if email in _beta_allowlist():
        base = f"{request.url.scheme}://{request.url.netloc}"
        token = _beta_sign("link", email, int(time.time()) + _BETA_LINK_TTL)
        if not _send_magic_link(email, f"{base}/?beta_token={urllib.parse.quote(token)}"):
            return {"ok": False, "error": "Could not send the email. Try again shortly."}
    return {"ok": True, "sent": True}


class BetaVerifyRequest(BaseModel):
    token: str


@app.post("/api/beta/verify")
def beta_verify(req: BetaVerifyRequest) -> JSONResponse:
    """Exchange a magic-link token for a session cookie."""
    email = _beta_verify("link", req.token)
    if email is None:
        return JSONResponse({"ok": False, "error": "This link is invalid or has expired."}, status_code=401)
    session = _beta_sign("session", email, int(time.time()) + _BETA_SESSION_TTL)
    resp = JSONResponse({"ok": True, "email": email})
    resp.set_cookie(
        _BETA_COOKIE,
        session,
        max_age=_BETA_SESSION_TTL,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    return resp


@app.post("/api/beta/logout")
def beta_logout() -> JSONResponse:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(_BETA_COOKIE, path="/")
    return resp


@app.get("/api/beta/status")
def beta_status(request: Request) -> dict:
    """Whether this browser may use the beta API (drives the unlock screen)."""
    email = _beta_session_email(request) if _IS_BETA else None
    return {
        "ok": True,
        "channel": _CHANNEL,
        "authorized": (not _IS_BETA) or email is not None,
        "email": email,
        "configured": bool(_BETA_SECRET and _RESEND_KEY and _BETA_EMAIL_FROM),
    }


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


class NameRequest(BaseModel):
    smiles: str
    include_trace: bool = False
    verify_opsin: bool = False
    token_debug: bool = False
    # Opt out of having this structure written to the result cache.
    no_cache: bool = False


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
            # include_trace=True mirrors the engine's verify branch, which
            # always analyzes; to_dict() below trims it when not requested.
            result = name_one(req.smiles, include_trace=True, token_debug=req.token_debug)
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


app.mount("/api", create_app())
