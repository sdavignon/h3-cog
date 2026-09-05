import json

import pytest

import h3_compliance
from h3_compliance import REQUIRED_ATTESTATIONS, enforce_moderation


def _values():
    return {
        "prompt": "A glass sculpture in morning light",
        "user_id": "user-test",
        "attestations": {name: True for name in REQUIRED_ATTESTATIONS},
    }


def test_moderation_fails_closed_without_configuration(monkeypatch):
    monkeypatch.delenv("H3_MODERATION_URL", raising=False)
    monkeypatch.delenv("H3_MODERATION_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="generation is blocked"):
        enforce_moderation(_values(), None, None)


def test_moderation_requires_all_attestations(monkeypatch):
    monkeypatch.setenv("H3_MODERATION_URL", "https://moderation.example/check")
    monkeypatch.setenv("H3_MODERATION_SECRET", "secret")
    monkeypatch.setattr(h3_compliance, "_public_https", lambda url: None)
    values = _values()
    values["attestations"]["likeness_rights_attested"] = False
    with pytest.raises(PermissionError, match="likeness_rights_attested"):
        enforce_moderation(values, None, None)


def test_moderation_accepts_authenticated_allow_decision(monkeypatch):
    monkeypatch.setenv("H3_MODERATION_URL", "https://moderation.example/check")
    monkeypatch.setenv("H3_MODERATION_SECRET", "secret")
    monkeypatch.setattr(h3_compliance, "_public_https", lambda url: None)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            return json.dumps(
                {"allow": True, "decision_id": "decision-1", "policy_version": "2026-09"}
            ).encode()

    monkeypatch.setattr(h3_compliance.urllib.request, "urlopen", lambda request, timeout: Response())
    assert enforce_moderation(_values(), None, None) == {
        "decision_id": "decision-1",
        "policy_version": "2026-09",
    }
