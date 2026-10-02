"""Hydrogenation scope is an assembly property, not a substring of its name."""

import pytest

from openclatura.assembler import _add_indicated_hydrogen_prefix
from openclatura.assembly_parts import AssemblyParts
from openclatura.name_operations import HydroOperation


@pytest.mark.parametrize("core,joiner", (("spiro[parent-1,1'-side]", ""), ("2'-azaspiro[parent-1,1'-side]", "-")))
def test_spiro_hydro_locants_cannot_be_elided_by_parent_atom_count(core, joiner):
    parts = AssemblyParts(parent_length=2)
    parts.hydro_operations.append(
        HydroOperation(
            key="additive_hydrogen",
            locants=("1", "2"),
            atom_ids=(7, 9),
            operation_kind="additive_hydrogen",
            reason="Graph-bound hydrogenation",
        )
    )
    assert _add_indicated_hydrogen_prefix(parts, core, allow_locant_elision=False) == f"1,2-dihydro{joiner}{core}"
