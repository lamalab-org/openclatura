#!/usr/bin/env python3
"""Report size and object count of the S3 result cache, by prefix.

Stdlib only. Reads the same env vars as the API:
    CACHE_S3_ENDPOINT / AWS_ENDPOINT_URL
    CACHE_S3_ACCESS_KEY / AWS_ACCESS_KEY_ID
    CACHE_S3_SECRET_KEY / AWS_SECRET_ACCESS_KEY
    CACHE_S3_BUCKET (default: openclatura-cache)

Usage:  python cache_size.py
"""

import datetime
import hashlib
import hmac
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

ENDPOINT = (os.environ.get("CACHE_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or "").rstrip("/")
ACCESS = os.environ.get("CACHE_S3_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID") or ""
SECRET = os.environ.get("CACHE_S3_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY") or ""
BUCKET = os.environ.get("CACHE_S3_BUCKET", "openclatura-cache")
REGION = os.environ.get("CACHE_S3_REGION", "us-east-1")


def s3_get(path: str, query: str) -> bytes:
    host = urllib.parse.urlparse(ENDPOINT).netloc
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date, datestamp = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(b"").hexdigest()
    headers = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
    signed = ";".join(sorted(headers))
    canonical = "\n".join(
        [
            "GET",
            urllib.parse.quote(path),
            query,
            "".join(f"{k}:{headers[k]}\n" for k in sorted(headers)),
            signed,
            payload_hash,
        ]
    )
    scope = f"{datestamp}/{REGION}/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    key = f"AWS4{SECRET}".encode()
    for part in (datestamp, REGION, "s3", "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(
        f"{ENDPOINT}{urllib.parse.quote(path)}?{query}",
        headers={
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
            "Authorization": f"AWS4-HMAC-SHA256 Credential={ACCESS}/{scope}, "
            f"SignedHeaders={signed}, Signature={signature}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace").strip()
        code = re.search(r"<Code>([^<]+)</Code>", body)
        message = re.search(r"<Message>([^<]+)</Message>", body)
        sys.exit(
            f"S3 {exc.code} {exc.reason}: {code.group(1) if code else '?'}"
            f" - {message.group(1) if message else body[:300] or '(empty body)'}\n"
            f"  endpoint={ENDPOINT} bucket={BUCKET} region={REGION} "
            f"access_key=...{ACCESS[-4:]} query={query}"
        )


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    if not (ENDPOINT and ACCESS and SECRET):
        sys.exit("Set CACHE_S3_ENDPOINT/ACCESS_KEY/SECRET_KEY (or AWS_*) first.")
    totals: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    token = ""
    while True:
        # SigV4 requires the canonical query string sorted by parameter name,
        # so continuation-token has to come before list-type.
        params = {"list-type": "2"}
        if token:
            params["continuation-token"] = token
        query = "&".join(f"{k}={urllib.parse.quote(params[k], safe='')}" for k in sorted(params))
        xml = s3_get(f"/{BUCKET}", query).decode()
        for key, size in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", xml, flags=re.S):
            prefix = "/".join(key.split("/")[:2])  # e.g. name/0.1.4
            totals[prefix][0] += 1
            totals[prefix][1] += int(size)
        match = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not match:
            break
        token = match.group(1)

    count = sum(int(v[0]) for v in totals.values())
    size = sum(v[1] for v in totals.values())
    print(f"{'prefix':<24}{'objects':>10}{'size':>12}")
    for prefix in sorted(totals):
        n, s = totals[prefix]
        print(f"{prefix:<24}{int(n):>10}{human(s):>12}")
    print("-" * 46)
    print(f"{'total':<24}{count:>10}{human(size):>12}")


if __name__ == "__main__":
    main()
