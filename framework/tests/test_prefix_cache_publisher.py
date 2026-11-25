# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest


@pytest.fixture
def publisher():
    script = Path(__file__).resolve().parents[2] / "scripts" / "publish-prefix-caches.py"
    spec = importlib.util.spec_from_file_location("prefix_cache_publisher", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(tmp_path, monkeypatch, suffix):
    fingerprint = ("prefix-cache 1|this model|old-build" + suffix).encode()
    prompt = struct.pack("<III", 0x4643504C, 1, len(fingerprint)) + fingerprint + struct.pack("<I", 96)
    (tmp_path / "prefix.prompt").write_bytes(prompt)
    entries = tmp_path / "entries.json"
    entries.write_text(json.dumps([{
        "file": "prefix.prompt", "engines": ["a" * 40], "os": "windows",
        "model": "model.gguf", "app": "claude-code", "app_version": "2.1.328",
    }]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["publish-prefix-caches.py", str(entries), "--dry-run"])
    return prompt


@pytest.mark.parametrize(("suffix", "message"), [
    ("", "lacks base-model provenance"),
    ("|adapter-cache=1|lora=0123456789abcdef", "computed with LoRA adapters"),
])
def test_rejects_unproven_and_adapted_files_before_contacting_service(
    publisher, tmp_path, monkeypatch, suffix, message,
):
    prepare(tmp_path, monkeypatch, suffix)
    monkeypatch.setattr(publisher, "call", lambda *_args: pytest.fail("unexpected release service request"))
    with pytest.raises(SystemExit, match=message):
        publisher.main()


def test_base_marker_allows_new_publication(publisher, tmp_path, monkeypatch, capsys):
    prompt = prepare(tmp_path, monkeypatch, "|adapter-cache=1")
    monkeypatch.setattr(publisher, "call", lambda *_args: (404, b""))
    publisher.main()
    name = f"prefix-cache-{hashlib.sha256(prompt).hexdigest()}.prompt"
    assert f"would upload {name} ({len(prompt)} bytes, 96 tokens)" in capsys.readouterr().out
