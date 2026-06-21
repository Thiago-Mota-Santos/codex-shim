from __future__ import annotations

import json
import time

import pytest

from codex_shim import claude_oauth
from codex_shim.server import (
    _anthropic_oauth_headers,
    _inject_claude_code_system,
    _merge_beta,
)
from codex_shim.settings import ModelSettings, byok_model_has_credentials


@pytest.fixture(autouse=True)
def isolate_oauth_env(monkeypatch, tmp_path):
    """Force every test onto a tmp credentials file with no env token."""
    monkeypatch.delenv(claude_oauth.CLAUDE_CODE_OAUTH_TOKEN_ENV, raising=False)
    monkeypatch.delenv("CODEX_SHIM_DISABLE_CLAUDE_OAUTH", raising=False)
    creds = tmp_path / ".credentials.json"
    monkeypatch.setenv(claude_oauth.CLAUDE_CREDENTIALS_ENV, str(creds))
    return creds


def _write_oauth(creds, *, access="oat-live", expires_at=None, refresh="ort-token"):
    if expires_at is None:
        expires_at = (time.time() * 1000) + 3_600_000
    creds.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": access,
                    "refreshToken": refresh,
                    "expiresAt": expires_at,
                }
            }
        )
    )


def test_available_false_when_nothing_present(isolate_oauth_env):
    assert claude_oauth.claude_oauth_available() is False


def test_env_token_takes_priority(monkeypatch, isolate_oauth_env):
    monkeypatch.setenv(claude_oauth.CLAUDE_CODE_OAUTH_TOKEN_ENV, "sk-ant-oat-env")
    assert claude_oauth.claude_oauth_available() is True
    assert claude_oauth.resolve_access_token() == "sk-ant-oat-env"


def test_disable_flag_overrides(monkeypatch, isolate_oauth_env):
    monkeypatch.setenv(claude_oauth.CLAUDE_CODE_OAUTH_TOKEN_ENV, "sk-ant-oat-env")
    monkeypatch.setenv("CODEX_SHIM_DISABLE_CLAUDE_OAUTH", "1")
    assert claude_oauth.claude_oauth_available() is False


def test_fresh_file_token_used_without_refresh(monkeypatch, isolate_oauth_env):
    _write_oauth(isolate_oauth_env, access="oat-fresh")

    def _boom(_refresh_token):
        raise AssertionError("refresh should not run for a fresh token")

    monkeypatch.setattr(claude_oauth, "refresh_oauth", _boom)
    assert claude_oauth.resolve_access_token() == "oat-fresh"


def test_expired_token_triggers_refresh_and_persists(monkeypatch, isolate_oauth_env):
    _write_oauth(isolate_oauth_env, access="oat-stale", expires_at=0, refresh="ort-1")
    captured = {}

    def _fake_refresh(refresh_token):
        captured["refresh_token"] = refresh_token
        return {"accessToken": "oat-new", "refreshToken": "ort-2", "expiresAt": (time.time() * 1000) + 3_600_000}

    monkeypatch.setattr(claude_oauth, "refresh_oauth", _fake_refresh)
    assert claude_oauth.resolve_access_token() == "oat-new"
    assert captured["refresh_token"] == "ort-1"

    persisted = json.loads(isolate_oauth_env.read_text())["claudeAiOauth"]
    assert persisted["accessToken"] == "oat-new"
    assert persisted["refreshToken"] == "ort-2"


def test_missing_token_raises(isolate_oauth_env):
    with pytest.raises(claude_oauth.ClaudeOAuthError):
        claude_oauth.resolve_access_token()


def test_oauth_model_has_credentials_via_env(monkeypatch, isolate_oauth_env, tmp_path):
    monkeypatch.setenv(claude_oauth.CLAUDE_CODE_OAUTH_TOKEN_ENV, "sk-ant-oat-env")
    settings = tmp_path / "models.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "claude-opus-4-8",
                        "provider": "anthropic",
                        "base_url": "https://api.anthropic.com",
                        "auth": "oauth",
                    }
                ]
            }
        )
    )
    model = ModelSettings(settings).load()[0]
    assert model.is_oauth is True
    assert model.api_key == ""
    assert byok_model_has_credentials(model) is True


def test_oauth_model_without_token_has_no_credentials(isolate_oauth_env, tmp_path):
    settings = tmp_path / "models.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "claude-opus-4-8",
                        "provider": "anthropic",
                        "base_url": "https://api.anthropic.com",
                        "auth": "oauth",
                    }
                ]
            }
        )
    )
    model = ModelSettings(settings).load()[0]
    assert byok_model_has_credentials(model) is False


def test_merge_beta_dedupes_and_appends():
    assert _merge_beta(None, "oauth-2025-04-20") == "oauth-2025-04-20"
    assert _merge_beta("oauth-2025-04-20", "oauth-2025-04-20") == "oauth-2025-04-20"
    assert _merge_beta("foo-1", "oauth-2025-04-20") == "foo-1, oauth-2025-04-20"


def test_oauth_headers_use_bearer_not_api_key(tmp_path):
    settings = tmp_path / "models.json"
    settings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "claude-opus-4-8",
                        "provider": "anthropic",
                        "base_url": "https://api.anthropic.com",
                        "auth": "oauth",
                        "extra_headers": {"x-api-key": "leak", "anthropic-beta": "foo-1"},
                    }
                ]
            }
        )
    )
    model = ModelSettings(settings).load()[0]
    headers = _anthropic_oauth_headers("tok-123", model)
    assert headers["Authorization"] == "Bearer tok-123"
    assert "x-api-key" not in headers
    assert headers["anthropic-beta"] == "foo-1, oauth-2025-04-20"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["User-Agent"].startswith("claude-")


def test_inject_system_prepends_spoof_for_string_system():
    out = _inject_claude_code_system({"system": "be terse", "model": "x"})
    assert out["system"][0]["text"] == claude_oauth.CLAUDE_CODE_SYSTEM_PROMPT
    assert out["system"][1]["text"] == "be terse"
    assert out["model"] == "x"


def test_inject_system_when_absent():
    out = _inject_claude_code_system({"model": "x"})
    assert out["system"] == [{"type": "text", "text": claude_oauth.CLAUDE_CODE_SYSTEM_PROMPT}]


def test_inject_system_is_idempotent_for_list():
    body = {"system": [{"type": "text", "text": claude_oauth.CLAUDE_CODE_SYSTEM_PROMPT}, {"type": "text", "text": "x"}]}
    assert _inject_claude_code_system(body) is body


def test_inject_system_does_not_mutate_input():
    body = {"system": [{"type": "text", "text": "original"}]}
    _inject_claude_code_system(body)
    assert body["system"] == [{"type": "text", "text": "original"}]
