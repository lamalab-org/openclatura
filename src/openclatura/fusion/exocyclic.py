"""Graph-bound external pi bonds supported by fused-parent composition."""

from ..hypervalent_roles import HypervalentLigandRole, hypervalent_center_role
from ..molecule import Molecule
from ..name_operations import AlkylideneOperation, IminoOperation, OxoOperation

ExternalPiOperation = OxoOperation | IminoOperation | AlkylideneOperation


def neutral_lambda_oxo_bonding_number(mol: Molecule, parent: int) -> int | None:
    """Prove a neutral chalcogen-oxo spectator without consuming skeletal pi.

    The two sigma bonds retain the ordinary skeletal valence; one or
    two terminal oxo ligands account for the entire lambda excess. Other
    hypervalent, charged, or internally multiply bonded sites are not proved.
    """
    atom = mol.atoms[parent]
    if (
        atom.element.standard_valence != 2
        or not atom.element.mancude_forced_single
        or atom.element.mancude_bonding_limit is not None
        or atom.charge
        or atom.total_h_count
        or atom.is_aromatic
    ):
        return None
    neighbors = mol.get_neighbors(parent)
    role = hypervalent_center_role(mol, {parent, *neighbors}, parent)
    if role is None:
        return None
    single = [ligand.atom for ligand in role.ligands if ligand.role == HypervalentLigandRole.SIGMA]
    oxo = [ligand.atom for ligand in role.ligands if ligand.role == HypervalentLigandRole.OXO]
    if (
        len(single) != atom.element.standard_valence
        or len(oxo) not in {1, 2}
        or len(single) + len(oxo) != len(neighbors)
    ):
        return None
    if any(mol.get_bond(parent, other).order != 1 for other in single):
        return None
    if any(
        mol.atoms[other].symbol != "O"
        or mol.atoms[other].charge
        or mol.atoms[other].total_h_count
        or len(mol.get_neighbors(other)) != 1
        or mol.get_bond(parent, other).order != 2
        for other in oxo
    ):
        return None
    return atom.element.standard_valence + 2 * len(oxo)


def is_neutral_external_pi_ligand(mol: Molecule, parent: int, ligand: int) -> bool:
    """Check a neutral external pi ligand without changing charge or H state."""
    atom = mol.atoms[ligand]
    bond = mol.get_bond(parent, ligand)
    return (
        (
            mol.atoms[parent].symbol == "C"
            or (atom.symbol == "O" and neutral_lambda_oxo_bonding_number(mol, parent) is not None)
        )
        and not mol.atoms[parent].charge
        and atom.symbol in {"O", "N", "C"}
        and not atom.charge
        and bond is not None
        and bond.order == 2
        and atom.total_h_count + sum(mol.get_bond(ligand, other).order for other in mol.get_neighbors(ligand))
        == atom.element.standard_valence
        and all(mol.get_bond(ligand, other).order == 1 for other in mol.get_neighbors(ligand) if other != parent)
    )
