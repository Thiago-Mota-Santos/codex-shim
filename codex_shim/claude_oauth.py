"""Claude subscription OAuth support for the Anthropic Messages upstream.

Lets an ``anthropic`` model in ``models.json`` authenticate with a Claude
Pro/Max subscription instead of a metered ``x-api-key``, mirroring the existing
ChatGPT/Cursor subscription passthroughs.

Credentials are read, in priority order:

1. ``CLAUDE_CODE_OAUTH_TOKEN`` — a long-lived token from ``claude setup-token``.
   Used verbatim, never refreshed.
2. ``~/.claude/.credentials.json`` — the file Claude Code writes after
   ``claude login``. Shape: ``{"claudeAiOauth": {"accessToken", "refreshToken",
   "expiresAt" (epoch ms), ...}}``. Short-lived access tokens (~60 min) are
   refreshed on demand against ``console.anthropic.com`` and written back.

The Anthropic API only accepts these tokens when the request looks like Claude
Code: ``Authorization: Bearer`` (not ``x-api-key``), the
``anthropic-beta: oauth-2025-04-20`` flag, and a first system block equal to
``CLAUDE_CODE_SYSTEM_PROMPT``. Header/body shaping lives in ``server.py``; this
module only resolves a fresh access token.
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_CLAUDE_CREDENTIALS = Path.home() / ".claude" / ".credentials.json"
CLAUDE_CODE_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
CLAUDE_CREDENTIALS_ENV = "CLAUDE_CODE_CREDENTIALS"
KEYCHAIN_SERVICE = "Claude Code-credentials"
KEYCHAIN_DISABLE_ENV = "CODEX_SHIM_DISABLE_KEYCHAIN"

CLAUDE_CODE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
CLAUDE_OAUTH_BETA = "oauth-2025-04-20"
CLAUDE_CODE_SYSTEM_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude."
CLAUDE_CODE_USER_AGENT = "claude-cli/2.0.0 (codex-shim)"

# Refresh a little early so a token never expires mid-flight.
EXPIRY_SKEW_MS = 60_000
REFRESH_TIMEOUT_SECONDS = 30.0


class ClaudeOAuthError(RuntimeError):
    """Raised when an OAuth token cannot be resolved or refreshed."""


def _credentials_path() -> Path:
    override = os.environ.get(CLAUDE_CREDENTIALS_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return DEFAULT_CLAUDE_CREDENTIALS


def _env_token() -> str:
    return os.environ.get(CLAUDE_CODE_OAUTH_TOKEN_ENV, "").strip()


def _read_credentials_file() -> dict[str, Any] | None:
    path = _credentials_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _keychain_enabled() -> bool:
    if sys.platform != "darwin":
        return False
    return os.environ.get(KEYCHAIN_DISABLE_ENV, "").lower() not in {"1", "true", "yes", "on"}


def _read_keychain() -> dict[str, Any] | None:
    """Read the Claude Code credential blob from the macOS login Keychain."""
    if not _keychain_enabled():
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _write_keychain(data: dict[str, Any]) -> bool:
    if not _keychain_enabled():
        return False
    try:
        subprocess.run(
            [
                "security",
                "add-generic-password",
                "-U",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                getpass.getuser(),
                "-w",
                json.dumps(data),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _oauth_from(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    oauth = data.get("claudeAiOauth")
    return oauth if isinstance(oauth, dict) and oauth.get("accessToken") else None


def _read_oauth_with_source() -> tuple[dict[str, Any], str] | None:
    """Find an OAuth block, preferring the credentials file, then the Keychain.

    Returns ``(oauth_block, source)`` where source is ``"file"`` or
    ``"keychain"`` so a refreshed token can be written back to the right place.
    """
    file_oauth = _oauth_from(_read_credentials_file())
    if file_oauth:
        return file_oauth, "file"
    keychain_oauth = _oauth_from(_read_keychain())
    if keychain_oauth:
        return keychain_oauth, "keychain"
    return None


def read_oauth() -> dict[str, Any] | None:
    """Return the ``claudeAiOauth`` block from file or Keychain, if present."""
    found = _read_oauth_with_source()
    return found[0] if found else None


def claude_oauth_available() -> bool:
    """True when a subscription token (env, file, or Keychain) is resolvable."""
    if os.environ.get("CODEX_SHIM_DISABLE_CLAUDE_OAUTH", "").lower() in {"1", "true", "yes", "on"}:
        return False
    if _env_token():
        return True
    return read_oauth() is not None


def _oauth_is_expired(oauth: dict[str, Any]) -> bool:
    expires_at = oauth.get("expiresAt")
    if not isinstance(expires_at, (int, float)):
        return True
    return (time.time() * 1000) >= (expires_at - EXPIRY_SKEW_MS)


def _persist_refreshed(oauth: dict[str, Any], source: str) -> None:
    if source == "keychain":
        data = _read_keychain() or {}
        data["claudeAiOauth"] = oauth
        _write_keychain(data)
        return
    path = _credentials_path()
    data = _read_credentials_file() or {}
    data["claudeAiOauth"] = oauth
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        path.chmod(0o600)
    except OSError:
        # A read-only credentials store is non-fatal: the refreshed token is
        # still returned for this run, just not cached for the next one.
        pass


def refresh_oauth(refresh_token: str) -> dict[str, Any]:
    """Exchange a refresh token for a fresh access token (blocking HTTP call)."""
    payload = json.dumps(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLAUDE_CODE_CLIENT_ID,
        }
    ).encode()
    request = urllib.request.Request(
        CLAUDE_OAUTH_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REFRESH_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace") if exc.fp else str(exc)
        raise ClaudeOAuthError(f"OAuth refresh failed ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        raise ClaudeOAuthError(f"OAuth refresh failed: {exc}") from exc

    access_token = body.get("access_token")
    if not access_token:
        raise ClaudeOAuthError("OAuth refresh response had no access_token")

    expires_in = body.get("expires_in")
    expires_at = (time.time() * 1000) + (float(expires_in) * 1000) if expires_in else None
    return {
        "accessToken": access_token,
        "refreshToken": body.get("refresh_token") or refresh_token,
        "expiresAt": expires_at,
    }


def resolve_access_token() -> str:
    """Return a usable Claude OAuth access token, refreshing the file if needed.

    Raises :class:`ClaudeOAuthError` when no token can be resolved. This makes a
    blocking refresh HTTP call when the cached token is expired, so callers on an
    event loop should run it in an executor.
    """
    env_token = _env_token()
    if env_token:
        return env_token

    found = _read_oauth_with_source()
    if not found:
        raise ClaudeOAuthError(
            f"No Claude OAuth token: set {CLAUDE_CODE_OAUTH_TOKEN_ENV}, run `claude login`, "
            "or run `claude setup-token`."
        )

    oauth, source = found
    if not _oauth_is_expired(oauth):
        return str(oauth["accessToken"])

    refresh_token = oauth.get("refreshToken")
    if not refresh_token:
        raise ClaudeOAuthError("Claude OAuth token expired and no refresh_token is available")

    refreshed = refresh_oauth(str(refresh_token))
    _persist_refreshed(refreshed, source)
    return str(refreshed["accessToken"])
