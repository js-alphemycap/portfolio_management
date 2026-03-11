"""HTTP helpers shared across scripts and library code."""

from __future__ import annotations

import os
import ssl
from pathlib import Path


def get_requests_verify() -> bool | str:
    """
    Return a `requests`-compatible `verify` value.

    This prefers an explicit CA bundle via environment variables, then certifi,
    and finally falls back to system default CA locations.
    """
    env_bundle = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("CURL_CA_BUNDLE")
    if env_bundle:
        return env_bundle

    try:  # pragma: no cover - environment dependent
        import certifi

        certifi_bundle = certifi.where()
        if certifi_bundle and Path(certifi_bundle).exists():
            return certifi_bundle
    except Exception:
        pass

    paths = ssl.get_default_verify_paths()
    candidates = [
        paths.cafile,
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    return True


__all__ = ["get_requests_verify"]

