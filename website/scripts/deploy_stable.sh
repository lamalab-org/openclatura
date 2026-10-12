#!/usr/bin/env bash
# Deploy the public (stable) site to the 'openclatura' Vercel project.
#
#   ./website/scripts/deploy_stable.sh              # version from pyproject.toml
#   ./website/scripts/deploy_stable.sh 0.4.0        # an explicit PyPI version
#
# CI runs this on every push to main that bumps the version (see
# .github/workflows/deploy-website.yml); it also works from a laptop after
# `npx vercel login`.
#
# The site installs openclatura at build time, so the deployment is pinned to
# exactly one version. The pin is taken from the argument or pyproject.toml,
# never from website/requirements.txt (whose openclatura line is replaced).
# If that version is not on PyPI yet the script waits for it, and if it still
# has not appeared after PYPI_WAIT_SECONDS it installs the package from the git
# commit instead (GIT_SHA, default: HEAD) so the release never stalls.
#
# Environment:
#   VERCEL_TOKEN, VERCEL_ORG_ID, VERCEL_PROJECT_ID   non-interactive auth (CI)
#   PYPI_WAIT_SECONDS   how long to wait for PyPI (default 1200; 0 = don't wait)
#   GIT_SHA             commit to install from when PyPI has no such version
set -euo pipefail

REPO_URL="https://github.com/lamalab-org/openclatura"
VERCEL="${VERCEL_BIN:-npx --yes vercel@latest}"

cd "$(dirname "$0")/.."
SITE_DIR="$PWD"
ROOT_DIR="$(cd .. && pwd)"

VERSION="${1:-$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$ROOT_DIR/pyproject.toml")}"
if [[ -z "$VERSION" ]]; then
  echo "error: could not read the version from pyproject.toml" >&2
  exit 1
fi
GIT_SHA="${GIT_SHA:-$(git -C "$ROOT_DIR" rev-parse HEAD)}"
WAIT="${PYPI_WAIT_SECONDS:-1200}"

on_pypi() {
  curl -fsS -o /dev/null "https://pypi.org/pypi/openclatura/$VERSION/json"
}

PIN="openclatura[web]==$VERSION"
if ! on_pypi; then
  echo "openclatura $VERSION is not on PyPI yet; waiting up to ${WAIT}s…"
  waited=0
  until on_pypi; do
    if (( waited >= WAIT )); then
      echo "still not on PyPI; installing from git commit $GIT_SHA instead."
      PIN="openclatura[web] @ git+${REPO_URL}@${GIT_SHA}"
      break
    fi
    sleep 30; waited=$((waited + 30))
  done
fi
echo "Pinning: $PIN"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Stage exactly what the deployment needs (same layout as deploy_beta.sh).
mkdir -p "$STAGE/api"
cp "$SITE_DIR/vercel.json" "$STAGE/"
cp -r "$SITE_DIR/public" "$STAGE/public"
find "$SITE_DIR/api" -maxdepth 1 -type f ! -name '*.pyc' -exec cp {} "$STAGE/api/" \;
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +

OTHER_DEPS="$(grep -v -e '^openclatura' -e '^#' -e '^$' "$SITE_DIR/requirements.txt" || true)"
{
  echo "# Staged by scripts/deploy_stable.sh — openclatura $VERSION (commit $GIT_SHA)."
  echo "$PIN"
  if [[ -n "$OTHER_DEPS" ]]; then printf '%s\n' "$OTHER_DEPS"; fi
} > "$STAGE/requirements.txt"

# With VERCEL_ORG_ID/VERCEL_PROJECT_ID in the environment (CI) the CLI needs no
# link; locally, reuse the stable project's link from website/.vercel.
if [[ -z "${VERCEL_PROJECT_ID:-}" && -d "$SITE_DIR/.vercel" ]]; then
  cp -r "$SITE_DIR/.vercel" "$STAGE/.vercel"
fi
TOKEN_ARGS=()
if [[ -n "${VERCEL_TOKEN:-}" ]]; then TOKEN_ARGS=(--token "$VERCEL_TOKEN"); fi

echo "Deploying openclatura $VERSION to production…"
$VERCEL deploy --prod --yes "${TOKEN_ARGS[@]}" --cwd "$STAGE"
