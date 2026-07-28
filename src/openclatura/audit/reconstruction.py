"""Structure-independent, OPSIN-free reconstruction audit.

The reconstruction audit is openclatura's *self-check*: it rebuilds the component
graph from the **name-level facts** the namer produced (parent length, retained
name, replacement prefixes, unsaturation locants, the principal-group key, and
substituent names) and compares the constitution to the input molecule.  It
never consults the namer's own atom→name bindings for the rebuild, so a bug in
the namer cannot mask itself here — the same soundness contract the substituent
reconstruction already follows.

Design principles:

* **Sound over complete.**  Anything the reconstructor does not fully model
  yields ``abstained`` (an honest "cannot certify"), never a guess and never a
  spurious ``mismatch``.  ``confirmed`` is only ever returned when a fully
  rebuilt graph matches the input.
* **Constitution here, stereo separately.**  The structural compare ignores
  stereochemistry; :func:`openclatura.audit.stereo.audit_stereochemistry`
  validates descriptors against graph metadata and is folded into the verdict.
* **Layered.**  The reconstruction is backed by always-applicable invariants —
  atom coverage (every heavy atom named) and charge-pair template support — so
  even when the rebuild abstains the audit still reports those checks.

The aggregate verdict is one of ``confirmed`` / ``abstained`` / ``mismatch`` /
``error``; see :class:`ReconstructionAudit`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rdkit import Chem

from ..molecule import Molecule
from ..rules import elements as _elements
from ..rules import multipliers as _multipliers
from .naming import (
    ChargePairTemplateAudit,
    NamingCoverage,
    audit_charge_pair_templates,
    component_named_atom_coverage,
)
from .relative_stereo import ring_face_relation
from .stereo import StereochemistryAudit, audit_stereochemistry
from .substituent_reconstruction import (
    _NAME_CIP,
    _NAME_RELATIVE,
    move_indicated_hydrogen,
    resolve_fragment_mol,
)
from .substituent_reconstruction import _RING_STEMS as _SUBSTITUENT_RING_STEMS
from .von_baeyer_parse import build_skeleton as _build_von_baeyer_skeleton
from .von_baeyer_parse import build_skeleton_from_descriptor as _build_von_baeyer_from_descriptor
from .von_baeyer_parse import build_spiro_skeleton as _build_spiro_skeleton

Verdict = Literal["confirmed", "abstained", "mismatch", "error"]


@dataclass(frozen=True)
class ReconstructionAudit:
    """Outcome of rebuilding a component from its name and comparing to input.

    ``verdict`` is the aggregate over the structural rebuild plus the coverage
    and stereo sub-audits:

    * ``confirmed`` — the graph was fully rebuilt from the name and matches the
      input, with no coverage or stereo problems.
    * ``abstained`` — some construct is not modelled (see ``reason``); the input
      is neither certified nor refuted.
    * ``mismatch`` — a positively rebuilt graph disagreed with the input, or an
      always-applicable invariant (unnamed atoms / stereo descriptor) failed.
    * ``error`` — an unexpected failure while auditing.
    """

    verdict: Verdict
    reason: str = ""
    reference_smiles: str | None = None
    reconstructed_smiles: str | None = None
    coverage: NamingCoverage | None = None
    stereo: StereochemistryAudit | None = None
    charge_pairs: ChargePairTemplateAudit | None = None

    @property
    def ok(self) -> bool:
        """Whether the name was positively certified against the input."""

        return self.verdict == "confirmed"

    @property
    def abstained(self) -> bool:
        return self.verdict == "abstained"

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        return self.verdict

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "reference_smiles": self.reference_smiles,
            "reconstructed_smiles": self.reconstructed_smiles,
            "unnamed_atoms": sorted(self.coverage.unnamed_atoms) if self.coverage else None,
            "stereo_issues": list(self.stereo.issues) if self.stereo else None,
            "unsupported_charge_pairs": (
                [
                    role.summary() if hasattr(role, "summary") else str(role)
                    for role in self.charge_pairs.unsupported_roles
                ]
                if self.charge_pairs
                else None
            ),
        }


# --------------------------------------------------------------------------- #
# Element / template tables (all name-derived, never graph-derived)
# --------------------------------------------------------------------------- #

# Replacement ("a") prefixes -> element symbol, shared with the von Baeyer parser
# and derived from the same element table the namer writes them from.
_REPLACEMENT_ELEMENTS: dict[str, str] = _elements.SYMBOLS_BY_HW_STEM

# Retained monocyclic parent rings whose IUPAC locants map straight onto the
# atom order of the SMILES below (heteroatom = locant 1).  Only rings whose
# numbering is unambiguous under this convention are listed; anything else
# abstains rather than risk a mislabelled locant.
_PARENT_RING_TEMPLATES: dict[str, tuple[str, list[str]]] = {
    "benzene": ("c1ccccc1", ["1", "2", "3", "4", "5", "6"]),
    "pyridine": ("n1ccccc1", ["1", "2", "3", "4", "5", "6"]),
    "pyrazine": ("n1ccncc1", ["1", "2", "3", "4", "5", "6"]),
    "pyrimidine": ("n1cnccc1", ["1", "2", "3", "4", "5", "6"]),
    "pyridazine": ("n1ncccc1", ["1", "2", "3", "4", "5", "6"]),
    "furan": ("o1cccc1", ["1", "2", "3", "4", "5"]),
    "thiophene": ("s1cccc1", ["1", "2", "3", "4", "5"]),
    "pyrrole": ("[nH]1cccc1", ["1", "2", "3", "4", "5"]),
    "piperidine": ("N1CCCCC1", ["1", "2", "3", "4", "5", "6"]),
    "piperazine": ("N1CCNCC1", ["1", "2", "3", "4", "5", "6"]),
    "morpholine": ("O1CCNCC1", ["1", "2", "3", "4", "5", "6"]),
    "pyrrolidine": ("N1CCCC1", ["1", "2", "3", "4", "5"]),
    "oxolane": ("O1CCCC1", ["1", "2", "3", "4", "5"]),
    "tetrahydrofuran": ("O1CCCC1", ["1", "2", "3", "4", "5"]),
    "oxane": ("O1CCCCC1", ["1", "2", "3", "4", "5", "6"]),
    "tetrahydropyran": ("O1CCCCC1", ["1", "2", "3", "4", "5", "6"]),
}

# Retained parents reuse the (OPSIN-validated) substituent ring-stem templates —
# same SMILES and IUPAC locant labels — so fused retained parents (naphthalene,
# indole, quinoline, benzothiazole …) reconstruct too.  Parent-specific entries
# win on any key overlap.
_ALL_PARENT_TEMPLATES: dict[str, tuple[str, list[str]]] = {
    **_SUBSTITUENT_RING_STEMS,
    **_PARENT_RING_TEMPLATES,
}


def _lookup_parent_template(retained_name: str) -> tuple[str, list[str]] | None:
    return _ALL_PARENT_TEMPLATES.get(retained_name)


# Principal characteristic groups whose structure we can rebuild with high
# confidence.  Each entry describes atoms bonded onto the *characteristic atom*
# as (element, bond_order) pairs.  ``exocyclic`` groups add a new carbon (bearing
# those atoms) attached to the parent locant atom; direct groups decorate the
# parent locant atom itself.  Every other key abstains.
_DIRECT_SUFFIX_GROUPS: dict[str, tuple[tuple[str, int], ...]] = {
    "alcohol": (("O", 1),),
    "thiol": (("S", 1),),
    "amine": (("N", 1),),
    "ketone": (("O", 2),),
    "aldehyde": (("O", 2),),
    "thioaldehyde": (("S", 2),),
    "imine": (("N", 2),),
    "carboxylic_acid": (("O", 2), ("O", 1)),
    "amide": (("O", 2), ("N", 1)),
    "thioamide": (("S", 2), ("N", 1)),
    "nitrile": (("N", 3),),
    "acid_chloride": (("O", 2), ("Cl", 1)),
    "acid_fluoride": (("O", 2), ("F", 1)),
    "acid_bromide": (("O", 2), ("Br", 1)),
    "acid_iodide": (("O", 2), ("I", 1)),
}
# Ring ("carb...") variants: a new carbon carries the same decoration.
_EXOCYCLIC_SUFFIX_GROUPS: dict[str, tuple[tuple[str, int], ...]] = {
    "ring_aldehyde": (("O", 2),),
    "ring_carboxylic_acid": (("O", 2), ("O", 1)),
    "ring_amide": (("O", 2), ("N", 1)),
    "ring_thioamide": (("S", 2), ("N", 1)),
    "ring_nitrile": (("N", 3),),
    "ring_acid_chloride": (("O", 2), ("Cl", 1)),
    "ring_acid_fluoride": (("O", 2), ("F", 1)),
    "ring_acid_bromide": (("O", 2), ("Br", 1)),
    "ring_acid_iodide": (("O", 2), ("I", 1)),
}
# Oxo-acid groups whose characteristic atom is a heteroatom *hub* bonded to the
# parent locant atom, itself decorated with the oxo/hydroxy atoms:
# ``R-S(=O)(=O)-OH`` etc.  Value: (hub element, decoration atoms).
_HUB_ACID_GROUPS: dict[str, tuple[str, tuple[tuple[str, int], ...]]] = {
    "sulfonic_acid": ("S", (("O", 2), ("O", 2), ("O", 1))),
    "sulfinic_acid": ("S", (("O", 2), ("O", 1))),
    "phosphonic_acid": ("P", (("O", 2), ("O", 1), ("O", 1))),
}

# Suffix groups whose characteristic atoms form a short chain rather than a set
# of atoms hung directly off the locant atom.  Value: SMILES of the added
# fragment, whose first atom bonds to the locant atom with the encoded order.
# ``ring_`` variants add their own carbon first, as the ``carb…`` suffixes do.
_FRAGMENT_SUFFIX_GROUPS: dict[str, str] = {
    "hydrazone": "=NN",
    "aldehyde_hydrazone": "=NN",
    "ring_aldehyde_hydrazone": "C=NN",
}

_BOND_TYPES: dict[int, Chem.BondType] = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


class _Abstain(Exception):
    """Internal signal: the reconstruction cannot model this construct."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def audit_component_reconstruction(
    mol: Molecule, parts, component_atoms: set[int] | None = None
) -> ReconstructionAudit:
    """Audit a named component by rebuilding it from ``parts`` and comparing.

    ``mol`` is the internal graph, ``parts`` the assembled :class:`AssemblyParts`
    for one component, and ``component_atoms`` the atom ids that component covers
    (defaults to the whole molecule).  Never raises.
    """

    try:
        atoms = set(component_atoms) if component_atoms is not None else set(mol.atoms)
        coverage = component_named_atom_coverage(mol, atoms, parts)
        charge_pairs = audit_charge_pair_templates(mol, atoms)

        reference = _component_reference_smiles(mol, atoms)
        structural_verdict, structural_reason, reconstructed, rebuilt = _structural_verdict(
            mol, atoms, parts, reference
        )

        # Stereo descriptors embedded in substituent terms are tagged onto the
        # rebuilt graph; verifying them against the input's independent CIP needs
        # the constitution to line up, so it runs after the structural rebuild.
        verified_atoms, verified_bonds = (
            _verify_tagged_stereo(rebuilt, mol, atoms) if rebuilt is not None else (set(), set())
        )
        stereo = audit_stereochemistry(mol, parts, atoms, verified_atoms, verified_bonds)

        # Only signals we can trust as *refutations* harden into ``mismatch``:
        #   * the independent structural reconstruction disagreed, or
        #   * the name provably left graph atoms unnamed.
        # Stereo problems, and anything the reconstruction could not model, only
        # *block confirmation* (they downgrade to ``abstained``) — never claim a
        # name is wrong on evidence too weak to prove it.
        hard_reasons: list[str] = []
        if structural_verdict == "mismatch":
            hard_reasons.append(structural_reason)
        if coverage.unnamed_atoms:
            details = ", ".join(f"{idx}:{mol.atoms[idx].symbol}" for idx in sorted(coverage.unnamed_atoms))
            hard_reasons.append(f"unnamed atoms: {details}")

        if hard_reasons:
            verdict: Verdict = "mismatch"
            reason = " | ".join(hard_reasons)
        elif structural_verdict == "confirmed" and stereo.ok:
            verdict = "confirmed"
            reason = ""
        elif structural_verdict == "confirmed":
            verdict = "abstained"
            reason = "constitution confirmed but stereo unverified: " + "; ".join(stereo.issues)
        else:
            verdict = "abstained"
            reason = structural_reason

        return ReconstructionAudit(
            verdict=verdict,
            reason=reason,
            reference_smiles=reference,
            reconstructed_smiles=reconstructed,
            coverage=coverage,
            stereo=stereo,
            charge_pairs=charge_pairs,
        )
    except Exception as exc:  # pragma: no cover - defensive; audit must never break naming
        return ReconstructionAudit(verdict="error", reason=f"{type(exc).__name__}: {exc}")


def _structural_verdict(mol, atoms, parts, reference) -> tuple[str, str, str | None, Chem.Mol | None]:
    """Return (verdict, reason, reconstructed_smiles, rebuilt_mol) for the rebuild."""

    if reference is None:
        return "abstained", "input constitution not comparable (charge/isotope/sanitize)", None, None
    if any(mol.atoms[a].isotope for a in atoms):
        return "abstained", "isotopically-labelled species not modelled", None, None
    # Net-charged (ionic) species are not modelled, but net-neutral charge
    # separation inside a group (nitro, N-oxide, azide, diazo …) is: those
    # charges are carried by the reconstructed fragment and compared like any
    # other constitution. Charged parents are still caught in _reconstruct_from_parts.
    if sum(mol.atoms[a].charge for a in atoms) != 0:
        return "abstained", "net-charged (ionic) species not modelled", None, None
    try:
        rebuilt = _reconstruct_from_parts(parts)
    except _Abstain as ab:
        return "abstained", ab.reason, None, None
    reconstructed = _canonical_constitution(rebuilt)
    if reconstructed is None:
        return "abstained", "reconstructed graph failed sanitisation", None, None
    if reconstructed == reference:
        return "confirmed", "", reconstructed, rebuilt
    return "mismatch", f"reconstructed {reconstructed!r} != input {reference!r}", reconstructed, rebuilt


# --------------------------------------------------------------------------- #
# Independent verification of name-asserted (substituent-embedded) stereo
# --------------------------------------------------------------------------- #
def _verify_tagged_stereo(rebuilt: Chem.Mol, mol: Molecule, atoms: set[int]) -> tuple[set[int], set[int]]:
    """Input atom ids and bond ids whose name-asserted stereo tag agrees with the
    input's independent CIP under *some* constitution isomorphism between the
    rebuilt graph and the input.

    The tags come from descriptors embedded in substituent terms (numbered in the
    substituent's own locant space).  Checking them against the independent CIP
    oracle needs a name→input mapping; the reconstruction is isomorphic to the
    input (constitution already confirmed), so any isomorphism under which every
    tag matches proves the name denotes the input's stereo *up to the molecule's
    own symmetry* — sound.  Untagged or unmatched features return nothing and are
    left for the caller to abstain on.

    Relative (``cis``/``trans``) tags ride along in the same pass: they are
    adjudicated against the input's tetrahedral parities rather than its CIP
    labels, but they bind to input atoms through the same isomorphism and must
    hold under the same one."""

    try:
        query = Chem.Mol(rebuilt)
        Chem.SanitizeMol(query)
    except Exception:
        return set(), set()
    Chem.RemoveStereochemistry(query)
    tagged_atoms = [(a.GetIdx(), a.GetProp(_NAME_CIP)) for a in query.GetAtoms() if a.HasProp(_NAME_CIP)]
    tagged_bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetProp(_NAME_CIP)) for b in query.GetBonds() if b.HasProp(_NAME_CIP)
    ]
    relative_pairs = _relative_stereo_pairs(query)
    if not tagged_atoms and not tagged_bonds and not relative_pairs:
        return set(), set()
    target, rev = _input_rdmol_with_cip(mol, atoms)
    if target is None:
        return set(), set()
    try:
        matches = target.GetSubstructMatches(query, uniquify=False, maxMatches=20000)
    except Exception:
        return set(), set()
    for match in matches:
        verified_atoms: set[int] = set()
        verified_bonds: set[int] = set()
        ok = True
        for qidx, descriptor in tagged_atoms:
            input_aid = rev.get(match[qidx])
            if input_aid is None or mol.atoms[input_aid].cip != descriptor:
                ok = False
                break
            verified_atoms.add(input_aid)
        if ok:
            for qa, qb, descriptor in tagged_bonds:
                ia, ib = rev.get(match[qa]), rev.get(match[qb])
                bond = mol.get_bond(ia, ib) if ia is not None and ib is not None else None
                if bond is None or bond.cip != descriptor:
                    ok = False
                    break
                verified_bonds.add(bond.idx)
        if ok:
            for qa, qb, word in relative_pairs:
                ia, ib = rev.get(match[qa]), rev.get(match[qb])
                if ia is None or ib is None or ring_face_relation(mol.audit_rdmol, ia, ib) != word:
                    ok = False
                    break
                verified_atoms.update((ia, ib))
        if ok:
            return verified_atoms, verified_bonds
    return set(), set()


def _relative_stereo_pairs(query: Chem.Mol) -> list[tuple[int, int, str]]:
    """The ``(atom, atom, "cis"|"trans")`` relations tagged on the rebuilt graph.

    A component can carry several independent relative assertions, so the tagged
    atoms are grouped by the ring they share rather than by the word — two
    different ``cis`` rings would otherwise be indistinguishable.  A group that
    is not exactly a pair is a relation we cannot pin down and is dropped, so the
    caller abstains on those centres."""

    tagged = {a.GetIdx(): a.GetProp(_NAME_RELATIVE) for a in query.GetAtoms() if a.HasProp(_NAME_RELATIVE)}
    if not tagged:
        return []
    pairs: list[tuple[int, int, str]] = []
    for ring in query.GetRingInfo().AtomRings():
        members = sorted(a for a in ring if a in tagged)
        if len(members) != 2:
            continue
        first, second = members
        if tagged[first] != tagged[second]:
            continue
        if any({first, second} == {a, b} for a, b, _ in pairs):
            continue  # a fused system reports the same pair through several rings
        pairs.append((first, second, tagged[first]))
    return pairs


def _input_rdmol_with_cip(mol: Molecule, atoms: set[int]) -> tuple[Chem.Mol | None, dict[int, int]]:
    """Build the input component as a stereo-free RDKit mol for isomorphism
    matching, returning it plus a ``rdkit_idx -> input_atom_id`` map."""
    rw = Chem.RWMol()
    idx_map: dict[int, int] = {}
    rev: dict[int, int] = {}
    for aid in sorted(atoms):
        atom = mol.atoms[aid]
        rd = Chem.Atom(atom.symbol)
        rd.SetFormalCharge(atom.charge)
        if atom.isotope:
            rd.SetIsotope(atom.isotope)
        j = rw.AddAtom(rd)
        idx_map[aid] = j
        rev[j] = aid
    for bond in mol.bonds.values():
        if bond.u in idx_map and bond.v in idx_map:
            rw.AddBond(idx_map[bond.u], idx_map[bond.v], _BOND_TYPES.get(bond.order, Chem.BondType.SINGLE))
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None, {}
    Chem.RemoveStereochemistry(m)
    return m, rev


# --------------------------------------------------------------------------- #
# Reference: the input component as canonical, stereo-free SMILES
# --------------------------------------------------------------------------- #
def _component_reference_smiles(mol: Molecule, atoms: set[int]) -> str | None:
    rw = Chem.RWMol()
    idx_map: dict[int, int] = {}
    for aid in sorted(atoms):
        atom = mol.atoms[aid]
        rdatom = Chem.Atom(atom.symbol)
        rdatom.SetFormalCharge(atom.charge)
        if atom.isotope:
            rdatom.SetIsotope(atom.isotope)
        idx_map[aid] = rw.AddAtom(rdatom)
    for bond in mol.bonds.values():
        if bond.u in idx_map and bond.v in idx_map:
            rw.AddBond(idx_map[bond.u], idx_map[bond.v], _BOND_TYPES.get(bond.order, Chem.BondType.SINGLE))
    return _canonical_constitution(rw.GetMol())


def _canonical_constitution(rwmol: Chem.Mol | None) -> str | None:
    if rwmol is None:
        return None
    mol = Chem.Mol(rwmol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    Chem.RemoveStereochemistry(mol)
    mol = _collapse_charge_separation(mol)
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _collapse_charge_separation(mol: Chem.Mol) -> Chem.Mol:
    """Rewrite ``[X+]-[O-]`` as ``X=O`` wherever the hypervalent spelling is legal.

    A sulfonyl can be drawn either way — ``S(=O)(=O)`` or ``[S+](=O)[O-]`` — and
    which one an input uses says nothing about the structure the name has to
    denote.  Both the input reference and the reconstruction pass through here,
    so the two spellings converge and no longer read as a disagreement.

    The rewrite is applied only when it survives sanitisation, which is what
    keeps it honest: groups whose charge separation is *not* optional (nitro,
    N-oxide, azide, diazo — nitrogen cannot take the extra bond) are rejected and
    keep their charges, on both sides alike.  Genuine differences in charge,
    element or connectivity survive untouched."""

    result = mol
    for _ in range(result.GetNumAtoms() + 1):  # each collapse removes one pair
        for bond in result.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
            if begin.GetFormalCharge() == 1 and end.GetFormalCharge() == -1:
                positive, negative = begin, end
            elif end.GetFormalCharge() == 1 and begin.GetFormalCharge() == -1:
                positive, negative = end, begin
            else:
                continue
            if negative.GetDegree() != 1:  # only a terminal counter-atom
                continue
            candidate = _promote_charge_pair(result, bond.GetIdx(), positive.GetIdx(), negative.GetIdx())
            if candidate is not None:
                result = candidate
                break
        else:
            break
    return result


def _promote_charge_pair(mol: Chem.Mol, bond_idx: int, positive_idx: int, negative_idx: int) -> Chem.Mol | None:
    """``[X+]-[Y-]`` -> ``X=Y`` with both charges cleared, or ``None`` when the
    charge separation was not an optional spelling after all.

    Two ways it can be rejected: the promoted form fails to sanitise, or
    sanitisation *restores* the charges — RDKit's clean-up rewrites a pentavalent
    ``N(=O)=O`` back to ``[N+](=O)[O-]``, which is precisely its way of saying
    that nitro has no neutral spelling.  Checking the charges actually stuck is
    also what makes this terminate."""

    rw = Chem.RWMol(mol)
    rw.GetBondWithIdx(bond_idx).SetBondType(Chem.BondType.DOUBLE)
    for idx in (positive_idx, negative_idx):
        atom = rw.GetAtomWithIdx(idx)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    candidate = rw.GetMol()
    try:
        Chem.SanitizeMol(candidate)
    except Exception:
        return None
    if any(candidate.GetAtomWithIdx(idx).GetFormalCharge() != 0 for idx in (positive_idx, negative_idx)):
        return None
    return candidate


# --------------------------------------------------------------------------- #
# Reconstruction from name-level parts
# --------------------------------------------------------------------------- #
def _reconstruct_from_parts(parts) -> Chem.RWMol:
    """Rebuild the component skeleton from ``parts``; raise ``_Abstain`` if any
    construct is not modelled."""

    has_template = parts.retained_name is not None and _lookup_parent_template(parts.retained_name) is not None
    if getattr(parts, "is_substituent", False):
        raise _Abstain("substituent component (audited via recursion, not here)")
    if parts.front_modifiers and not _is_ester_component(parts):
        # Front modifiers we model are ester/sulfonate esterifying groups; other
        # uses are not reconstructed.
        raise _Abstain("front modifiers not modelled")
    if parts.parent_charges:
        raise _Abstain("parent charges not modelled")
    if parts.indicated_hydrogens and (not has_template or len(parts.indicated_hydrogens) != 1):
        # Templates place their indicated H at position 1 (1H-pyrrole -> [nH]).
        # A single cited position elsewhere (2H-indazole, 9H-purine) is another
        # N-H tautomer, which is reachable by moving that hydrogen; several at
        # once is not modelled.
        raise _Abstain("indicated hydrogen position not modelled")
    if any(op.operation_kind == "additive_hydrogen" for op in parts.hydro_operations):
        raise _Abstain("added (hydro) hydrogen not modelled")
    if parts.principal_suffix_modifiers:
        raise _Abstain("principal-suffix modifiers not modelled")

    rw, locants, aromatic_ring = _build_parent(parts)
    if parts.indicated_hydrogens and not move_indicated_hydrogen(rw, locants, parts.indicated_hydrogens[0]):
        raise _Abstain(f"indicated hydrogen {parts.indicated_hydrogens[0]} not placeable")
    _apply_replacements(rw, locants, parts, aromatic_ring)
    _apply_unsaturations(rw, locants, parts, aromatic_ring)
    _apply_principal_group(rw, locants, parts)
    _apply_substituents(rw, locants, parts)
    return rw


def _build_parent(parts) -> tuple[Chem.RWMol, dict[str, int], bool]:
    """Return (editable mol, locant->idx, is_aromatic_template)."""

    # Retained parents (mono- or fused) reconstruct from their template first.
    if parts.retained_name is not None:
        template = _lookup_parent_template(parts.retained_name)
        if template is None:
            raise _Abstain(f"retained parent {parts.retained_name!r} not modelled")
        smiles, labels = template
        frag = Chem.MolFromSmiles(smiles)
        if frag is None or frag.GetNumAtoms() != len(labels):
            raise _Abstain("retained template failed to parse")
        rw = Chem.RWMol(frag)
        aromatic = any(a.GetIsAromatic() for a in frag.GetAtoms())
        return rw, {label: i for i, label in enumerate(labels)}, aromatic

    if parts.is_spiro:
        a, b = parts.spiro_xy
        rw, locants = _build_spiro_skeleton(a, b)
        if rw is None or a + b + 1 != parts.parent_length:
            raise _Abstain(f"spiro skeleton {parts.spiro_xy} inconsistent")
        return rw, locants, False

    if parts.is_bicycle:
        a, b, c = parts.bicycle_xyz
        rw, locants = _build_von_baeyer_skeleton(a, b, c)
        if rw is None or a + b + c + 2 != parts.parent_length:
            raise _Abstain(f"von Baeyer skeleton {parts.bicycle_xyz} inconsistent")
        return rw, locants, False

    if parts.is_polycycle:
        rw, locants = _build_von_baeyer_from_descriptor(parts.polycycle_descriptor or "")
        if rw is None or len(locants) != parts.parent_length:
            raise _Abstain(f"polycyclic descriptor {parts.polycycle_descriptor!r} not modelled")
        return rw, locants, False

    if parts.is_ring:
        # plain carbocycle
        n = parts.parent_length
        if n < 3:
            raise _Abstain("degenerate ring size")
        rw = Chem.RWMol()
        idxs = [rw.AddAtom(Chem.Atom(6)) for _ in range(n)]
        for a, b in zip(idxs, idxs[1:]):
            rw.AddBond(a, b, Chem.BondType.SINGLE)
        rw.AddBond(idxs[-1], idxs[0], Chem.BondType.SINGLE)
        return rw, {str(i + 1): idxs[i] for i in range(n)}, False

    if parts.retained_name is not None:
        raise _Abstain(f"retained acyclic parent {parts.retained_name!r} not modelled")
    n = parts.parent_length
    if n < 1:
        raise _Abstain("empty parent chain")
    rw = Chem.RWMol()
    idxs = [rw.AddAtom(Chem.Atom(6)) for _ in range(n)]
    for a, b in zip(idxs, idxs[1:]):
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    return rw, {str(i + 1): idxs[i] for i in range(n)}, False


_LAMBDA_RE = re.compile(r"lambda\^?\{?\d+\}?")


def _apply_replacements(rw: Chem.RWMol, locants: dict[str, int], parts, aromatic_ring: bool) -> None:
    for item in parts.a_prefixes:
        element = _replacement_element(item.name)
        if element is None:
            raise _Abstain(f"replacement prefix {item.name!r} not modelled")
        if aromatic_ring:
            raise _Abstain("replacement on a retained aromatic template not modelled")
        for locant in item.locants:
            # ``1lambda^6`` is position 1 wearing a lambda-convention valence; the
            # extra bonds come from the oxo prefixes grafted later, so only the
            # position matters here.
            idx = locants.get(_LAMBDA_RE.sub("", str(locant)))
            if idx is None:
                raise _Abstain(f"replacement locant {locant} outside parent")
            rw.GetAtomWithIdx(idx).SetAtomicNum(Chem.Atom(element).GetAtomicNum())


def _replacement_element(name: str) -> str | None:
    if name in _REPLACEMENT_ELEMENTS:
        return _REPLACEMENT_ELEMENTS[name]
    # ``trioxa`` names the same element as ``oxa``; how many there are comes from
    # the cited locants, so only the element is read back here.
    for _count, rest in _multipliers.candidate_splits(name):
        if rest in _REPLACEMENT_ELEMENTS:
            return _REPLACEMENT_ELEMENTS[rest]
    return None


def _apply_unsaturations(rw: Chem.RWMol, locants: dict[str, int], parts, aromatic_ring: bool) -> None:
    for item in parts.unsaturations:
        order = 2 if item.bond_key == "double" else 3 if item.bond_key == "triple" else None
        if order is None:
            raise _Abstain(f"unsaturation {item.bond_key!r} not modelled")
        if aromatic_ring:
            raise _Abstain("explicit unsaturation on aromatic template not modelled")
        for locant in item.locants:
            lo, hi = _parse_unsaturation_locant(str(locant))
            if lo is None:
                raise _Abstain(f"unsaturation locant {locant!r} not modelled")
            a_idx = locants.get(lo)
            b_idx = locants.get(hi) if hi is not None else _next_locant_idx(locants, lo, parts.is_ring)
            if a_idx is None or b_idx is None:
                raise _Abstain(f"unsaturation locant {locant} has no partner")
            bond = rw.GetBondBetweenAtoms(a_idx, b_idx)
            if bond is None:
                raise _Abstain(f"no parent bond at unsaturation locant {locant}")
            bond.SetBondType(_BOND_TYPES[order])


def _parse_unsaturation_locant(token: str) -> tuple[str | None, str | None]:
    """Split an unsaturation locant into (start, explicit-partner). ``4`` -> the
    bond 4→5 (partner ``None``); ``1(10)`` -> the fusion bond 1→10."""
    m = re.fullmatch(r"(\d+)(?:\((\d+)\))?", token)
    if m is None:
        return None, None
    return m.group(1), m.group(2)


def _next_locant_idx(locants: dict[str, int], locant: str, is_ring: bool) -> int | None:
    try:
        nxt = str(int(locant) + 1)
    except ValueError:
        return None
    if nxt in locants:
        return locants[nxt]
    if is_ring:  # wrap N -> 1
        return locants.get("1")
    return None


# Ester-type principal groups: an esterifying group (the "ethyl" of "ethyl
# benzoate") is carried as a front modifier and closes the acid as -O-R.
#   direct    : acid carbon is in-chain    -> C(=O)-O-R at the locant atom
#   exocyclic : acid carbon is added        -> ring-C(=O)-O-R
#   sulfonate : exocyclic S(=O)(=O)-O-R
_ESTER_DIRECT = {"ester", "carboxylate"}
_ESTER_EXOCYCLIC = {"ring_carboxylate"}
_ESTER_SULFONATE = {"sulfonate"}
_ESTER_KEYS = _ESTER_DIRECT | _ESTER_EXOCYCLIC | _ESTER_SULFONATE


def _is_ester_component(parts) -> bool:
    pg = parts.principal_group
    return pg is not None and pg.key in _ESTER_KEYS


def _apply_principal_group(rw: Chem.RWMol, locants: dict[str, int], parts) -> None:
    pg = parts.principal_group
    if pg is None:
        if parts.front_modifiers:
            raise _Abstain("front modifiers without a principal group")
        return
    if pg.key in _ESTER_KEYS:
        _apply_ester(rw, locants, parts)
        return
    if pg.key in _HUB_ACID_GROUPS:
        hub_element, decoration = _HUB_ACID_GROUPS[pg.key]
        for locant in pg.locants:
            base_idx = locants.get(str(locant))
            if base_idx is None:
                raise _Abstain(f"principal-group locant {locant} outside parent")
            hub = rw.AddAtom(Chem.Atom(hub_element))
            rw.AddBond(base_idx, hub, Chem.BondType.SINGLE)
            _decorate(rw, hub, decoration)
        return
    fragment = _FRAGMENT_SUFFIX_GROUPS.get(pg.key)
    if fragment is not None:
        added = Chem.MolFromSmiles("*" + fragment)
        if added is None:
            raise _Abstain(f"principal group {pg.key!r} failed to parse")
        terminal_nitrogens: list[int] = []
        for locant in pg.locants:
            base_idx = locants.get(str(locant))
            if base_idx is None:
                raise _Abstain(f"principal-group locant {locant} outside parent")
            first_new = rw.GetNumAtoms()
            _graft(rw, base_idx, added)
            # A hydrazone's ``N`` locant is its *terminal* nitrogen — ``benzaldehyde
            # N-phenylhydrazone`` is ``PhNH-N=CHPh`` — and that is the only nitrogen
            # the graft leaves with a single neighbour.
            terminal_nitrogens += [
                idx
                for idx in range(first_new, rw.GetNumAtoms())
                if rw.GetAtomWithIdx(idx).GetAtomicNum() == 7 and rw.GetAtomWithIdx(idx).GetDegree() == 1
            ]
        _expose_n_locants(locants, terminal_nitrogens)
        return
    direct = _DIRECT_SUFFIX_GROUPS.get(pg.key)
    exocyclic = _EXOCYCLIC_SUFFIX_GROUPS.get(pg.key)
    if direct is None and exocyclic is None:
        raise _Abstain(f"principal group {pg.key!r} not modelled")
    nitrogens: list[int] = []
    for locant in pg.locants:
        base_idx = locants.get(str(locant))
        if base_idx is None:
            raise _Abstain(f"principal-group locant {locant} outside parent")
        if exocyclic is not None:
            carbon = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(base_idx, carbon, Chem.BondType.SINGLE)
            nitrogens += _decorate(rw, carbon, exocyclic)
        else:
            nitrogens += _decorate(rw, base_idx, direct)
    _expose_n_locants(locants, nitrogens)


def _expose_n_locants(locants: dict[str, int], nitrogens: list[int]) -> None:
    """Register a suffix's characteristic nitrogens under the italic ``N`` locants
    so that N-substituents (N-methyl, N,N-dimethyl, N'-ethyl…) can attach.

    A single one is just ``N``; several are primed — ``N``, ``N'``, ``N''`` — in
    the order their parent locants were cited, matching IUPAC's prime
    assignment."""

    for i, nitrogen in enumerate(nitrogens):
        key = "N" + "'" * i
        if key not in locants:
            locants[key] = nitrogen


def _apply_ester(rw: Chem.RWMol, locants: dict[str, int], parts) -> None:
    pg = parts.principal_group
    mods = list(parts.front_modifiers)
    mod_locs = [str(loc) for loc in parts.front_modifier_locants]
    if len(mod_locs) != len(mods):
        mod_locs = [str(loc) for loc in pg.locants]
    if len(mods) != len(mod_locs):
        raise _Abstain("ester front-modifier/locant count mismatch")
    for name, loc in zip(mods, mod_locs):
        r_frag = resolve_fragment_mol(name) if name else None
        if r_frag is None:
            raise _Abstain(f"ester group {name!r} not modelled")
        base_idx = locants.get(loc)
        if base_idx is None:
            raise _Abstain(f"ester locant {loc} outside parent")
        _build_ester_group(rw, base_idx, pg.key, r_frag)


def _build_ester_group(rw: Chem.RWMol, base_idx: int, key: str, r_frag: Chem.Mol) -> None:
    if key in _ESTER_SULFONATE:
        sulfur = rw.AddAtom(Chem.Atom(16))
        rw.AddBond(base_idx, sulfur, Chem.BondType.SINGLE)
        for _ in range(2):
            oxo = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(sulfur, oxo, Chem.BondType.DOUBLE)
        ester_o = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(sulfur, ester_o, Chem.BondType.SINGLE)
        _graft(rw, ester_o, r_frag)
        return
    acid_c = base_idx
    if key in _ESTER_EXOCYCLIC:
        acid_c = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(base_idx, acid_c, Chem.BondType.SINGLE)
    oxo = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(acid_c, oxo, Chem.BondType.DOUBLE)
    ester_o = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(acid_c, ester_o, Chem.BondType.SINGLE)
    _graft(rw, ester_o, r_frag)


def _decorate(rw: Chem.RWMol, base_idx: int, atoms: tuple[tuple[str, int], ...]) -> list[int]:
    """Add ``atoms`` onto ``base_idx``; return the indices of any added nitrogens."""
    nitrogens: list[int] = []
    for element, order in atoms:
        new = rw.AddAtom(Chem.Atom(element))
        rw.AddBond(base_idx, new, _BOND_TYPES[order])
        if element == "N":
            nitrogens.append(new)
    return nitrogens


# Side-ring suffixes we can express as an equivalent prefix on the ring.
_SPIRO_SUFFIX_PREFIXES = {"one": "oxo", "ol": "hydroxy", "thione": "thioxo"}


def _spiro_side_name(spiro) -> str | None:
    """Rewrite a spiro side ring as an ordinary ``-yl`` substituent name.

    A spiro assembly cites its side ring in parent form (``cyclopentane``,
    ``bicyclo[4.3.0]nona-1,3,5-triene``) with primed locants, so it is turned
    into ``<prefixes><stem>-<side locant>-yl`` and handed to the substituent
    grammar — which already knows every ring family that can appear here.  The
    primes only distinguish the side ring's locants from the parent's and are
    dropped; a side suffix becomes its equivalent prefix (``4'-one`` ->
    ``4-oxo``).  ``None`` for anything outside that rewrite."""

    clauses = [prefix.replace("'", "") for prefix in spiro.side_prefixes]
    for locant, suffix in spiro.side_suffixes:
        prefix = _SPIRO_SUFFIX_PREFIXES.get(suffix)
        if prefix is None:
            return None
        clauses.append(f"{str(locant).replace(chr(39), '')}-{prefix}")
    stem = spiro.side_parent_name
    if stem.endswith("e"):  # ``cyclopentane`` -> ``cyclopentan-2-yl``
        stem = stem[:-1]
    return "-".join([*clauses, f"{stem}-{str(spiro.side_locant).replace(chr(39), '')}-yl"])


def _fuse_spiro(rw: Chem.RWMol, base_idx: int, frag: Chem.Mol) -> None:
    """Merge a side ring onto the parent at its spiro atom.

    The rewritten side ring arrives as an ordinary fragment whose dummy sits on
    the spiro atom.  Spiro fusion *identifies* that atom with the parent's rather
    than bonding to it, so the spiro atom is dropped and its two ring bonds are
    re-made from the parent atom — which then carries four ring bonds, two per
    ring, as a spiro atom must."""

    dummy = next((a for a in frag.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dummy is None or len(dummy.GetNeighbors()) != 1:
        raise _Abstain("spiro side ring has no single attachment point")
    shared = dummy.GetNeighbors()[0]
    if not shared.IsInRing():
        # A spiro union joins two *rings* at a shared atom.  A side that resolves
        # to a chain means the name is not describing a spiro assembly we can
        # rebuild, and fusing it anyway would invent a structure to compare
        # against — so abstain rather than manufacture a disagreement.
        raise _Abstain("spiro side is not a ring")
    dropped = {dummy.GetIdx(), shared.GetIdx()}
    frag_to_new: dict[int, int] = {}
    for atom in frag.GetAtoms():
        if atom.GetIdx() in dropped:
            continue
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetNoImplicit(atom.GetNoImplicit())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        frag_to_new[atom.GetIdx()] = rw.AddAtom(new_atom)
    for bond in frag.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if dummy.GetIdx() in (a, b):
            continue
        if shared.GetIdx() in (a, b):  # a ring bond of the shared atom, re-made from the parent
            other = b if a == shared.GetIdx() else a
            _consume_parent_hydrogen(rw.GetAtomWithIdx(base_idx), bond.GetBondType())
            rw.AddBond(base_idx, frag_to_new[other], bond.GetBondType())
        else:
            rw.AddBond(frag_to_new[a], frag_to_new[b], bond.GetBondType())


def _apply_spiro_substituent(rw: Chem.RWMol, locants: dict[str, int], item) -> None:
    base_idx = locants.get(str(item.spiro.parent_locant))
    if base_idx is None:
        raise _Abstain(f"spiro locant {item.spiro.parent_locant} outside parent")
    side_name = _spiro_side_name(item.spiro)
    frag = resolve_fragment_mol(side_name) if side_name else None
    if frag is None:
        raise _Abstain(f"spiro side ring {item.spiro.side_parent_name!r} not modelled")
    _fuse_spiro(rw, base_idx, frag)


def _apply_substituents(rw: Chem.RWMol, locants: dict[str, int], parts) -> None:
    for item in parts.substituents:
        if item.spiro is not None:
            _apply_spiro_substituent(rw, locants, item)
            continue
        frag = resolve_fragment_mol(item.name)
        if frag is None:
            raise _Abstain(f"substituent {item.name!r} not modelled")
        for locant in item.locants:
            base_idx = locants.get(str(locant))
            if base_idx is None:
                raise _Abstain(f"substituent locant {locant} outside parent")
            _graft(rw, base_idx, frag)


def _graft(rw: Chem.RWMol, base_idx: int, frag: Chem.Mol) -> None:
    dummy = next((a for a in frag.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dummy is None:
        raise _Abstain("substituent fragment lacks an attachment point")
    frag_to_new: dict[int, int] = {}
    for atom in frag.GetAtoms():
        if atom.GetIdx() == dummy.GetIdx():
            continue
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetNoImplicit(atom.GetNoImplicit())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        for prop in (_NAME_CIP, _NAME_RELATIVE):  # carry name-asserted stereo tags onto the assembled graph
            if atom.HasProp(prop):
                new_atom.SetProp(prop, atom.GetProp(prop))
        frag_to_new[atom.GetIdx()] = rw.AddAtom(new_atom)
    for bond in frag.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if dummy.GetIdx() in (a, b):
            other = b if a == dummy.GetIdx() else a
            _consume_parent_hydrogen(rw.GetAtomWithIdx(base_idx), bond.GetBondType())
            rw.AddBond(base_idx, frag_to_new[other], bond.GetBondType())
            _carry_bond_tag(bond, rw.GetBondBetweenAtoms(base_idx, frag_to_new[other]))
        else:
            rw.AddBond(frag_to_new[a], frag_to_new[b], bond.GetBondType())
            _carry_bond_tag(bond, rw.GetBondBetweenAtoms(frag_to_new[a], frag_to_new[b]))


def _carry_bond_tag(src: Chem.Bond, dst: Chem.Bond | None) -> None:
    if dst is not None and src.HasProp(_NAME_CIP):
        dst.SetProp(_NAME_CIP, src.GetProp(_NAME_CIP))


def _consume_parent_hydrogen(atom: Chem.Atom, bond_type: Chem.BondType) -> None:
    """Substituting at a ring atom carrying an explicit H (e.g. indole N1) uses
    that hydrogen; implicit-H atoms are left for RDKit to rebalance."""
    explicit = atom.GetNumExplicitHs()
    if explicit <= 0:
        return
    order = 2 if bond_type == Chem.BondType.DOUBLE else 3 if bond_type == Chem.BondType.TRIPLE else 1
    atom.SetNumExplicitHs(max(0, explicit - order))


__all__ = ["ReconstructionAudit", "audit_component_reconstruction"]
