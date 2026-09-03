import subprocess
import sys


def test_legacy_naming_does_not_load_systematic_fusion_engine():
    program = """
import sys
import openclatura

heavy_modules = {
    "openclatura.fusion.audit",
    "openclatura.fusion.cover",
    "openclatura.fusion.descriptor",
    "openclatura.fusion.faces",
    "openclatura.fusion.layout",
    "openclatura.fusion.numbering",
    "openclatura.fusion.planner",
    "openclatura.fusion.registry",
}
assert not heavy_modules.intersection(sys.modules)
result = openclatura.name("CCO")
assert result.error is None
assert not heavy_modules.intersection(sys.modules)
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_fusion_public_facade_keeps_lazy_exports_available():
    from openclatura.fusion import FusionMode, render_fusion_name_parts

    assert FusionMode.LEGACY.value == "legacy"
    assert callable(render_fusion_name_parts)
