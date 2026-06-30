# syntax=docker/dockerfile:1.7

# ----- builder stage -----------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build deps for rdkit wheels: typically just runtime; the manylinux wheels
# carry their own libs. Keep this stage thin.
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Install into a clean prefix so we can copy a slim layer into the runtime.
RUN python -m pip install --upgrade pip \
 && python -m pip install --prefix=/install ".[web,opsin]"


# ----- runtime stage -----------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/bin:${PATH}"

# Java runtime for OPSIN round-trip verification (py2opsin shells out to java;
# OPSIN itself requires Java >=8 so the newer JRE is fine).
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends \
        default-jre-headless \
        curl \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Bring in the installed package + deps from the builder stage.
COPY --from=builder /install /usr/local

# Non-root user.
RUN useradd --create-home --shell /bin/bash openclatura
USER openclatura
WORKDIR /home/openclatura

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["python", "-m", "openclatura.web", "--host", "0.0.0.0", "--port", "8000"]
