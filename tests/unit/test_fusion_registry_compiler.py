import subprocess
import sys
from pathlib import Path


def test_generated_fusion_component_registry_is_current():
    root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [sys.executable, str(root / "scripts/build_fusion_component_registry.py"), "--check"],
        cwd=root,
        check=True,
    )
