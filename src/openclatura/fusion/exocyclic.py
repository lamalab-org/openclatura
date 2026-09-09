"""Graph-bound external pi bonds supported by fused-parent composition."""

from ..molecule import Molecule
from ..name_operations import IminoOperation, OxoOperation

ExternalPiOperation = OxoOperation | IminoOperation


def is_neutral_external_pi_ligand(mol: Molecule, parent: int, ligand: int) -> bool:
    """Check a carbonyl or imino ligand without changing its charge or H state."""
    atom = mol.atoms[ligand]
    bond = mol.get_bond(parent, ligand)
    return (
        mol.atoms[parent].symbol == "C"
        and not mol.atoms[parent].charge
        and atom.symbol in {"O", "N"}
        and not atom.charge
        and bond is not None
        and bond.order == 2
        and atom.total_h_count + sum(mol.get_bond(ligand, other).order for other in mol.get_neighbors(ligand))
        == atom.element.standard_valence
        and all(mol.get_bond(ligand, other).order == 1 for other in mol.get_neighbors(ligand) if other != parent)
    )
