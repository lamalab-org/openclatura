import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from openclatura import opsin_verify as ov


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("stderr", [b"OPSIN failed\xff", "OPSIN failed", None])
def test_subprocess_errors_preserve_diagnostics(monkeypatch, masked, stderr):
    paths = []

    def failing_opsin(names, *, tmp_fpath):
        assert names == ["ethanol"]
        paths.append(Path(tmp_fpath))
        paths[-1].write_text(names[0])
        try:
            raise subprocess.CalledProcessError(1, ["java", "-jar", "opsin.jar"], stderr=stderr)
        except subprocess.CalledProcessError as exc:
            if masked:
                # Reproduce py2opsin's broken warning construction.
                "Unexpected error ocurred! " + exc
            raise

    monkeypatch.setattr(ov, "_try_import_py2opsin", lambda: SimpleNamespace(py2opsin=failing_opsin))
    monkeypatch.setattr(ov, "_java_available", lambda: True)

    check = ov.verify_with_opsin("ethanol", "CCO")

    assert check.status == "error"
    assert not check.ok
    assert check.name == "ethanol"
    assert check.canonical_original == "CCO"
    assert "CalledProcessError" in check.error_message
    assert "exit status 1" in check.error_message
    assert ("TypeError" in check.error_message) == masked
    if stderr:
        assert "OPSIN stderr: OPSIN failed" in check.error_message
    assert not paths[0].parent.exists()


def test_generic_opsin_error_is_structured(monkeypatch):
    def failing_opsin(*args, **kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(ov, "_try_import_py2opsin", lambda: SimpleNamespace(py2opsin=failing_opsin))
    monkeypatch.setattr(ov, "_java_available", lambda: True)
    check = ov.verify_with_opsin("ethanol", "CCO")
    assert check.status == "error"
    assert check.error_message == "RuntimeError: unexpected failure"


def test_error_message_handles_cyclic_exception_context():
    exc = RuntimeError("failure")
    exc.__context__ = exc
    assert ov._opsin_error_message(exc) == "RuntimeError: failure"
