# structure-to-iupac/perception.py

from dataclasses import dataclass
from .molecule import Molecule
from .chains import get_cyclic_atoms

@dataclass
class PerceivedGroup:
    key: str
    is_principal_candidate: bool
    attachment_carbon: int
    atoms_involved: set[int]

def perceive_groups(mol: Molecule) -> list[PerceivedGroup]:
    groups =[]
    consumed = set()
    cyclic_atoms = get_cyclic_atoms(mol)

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            oxygens =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "O"]
            adj_atoms =[n for n in mol.get_neighbors(atom.idx) if n not in oxygens]
            
            if len(oxygens) == 2 and len(adj_atoms) == 1:
                has_double_o = any(mol.get_bond(atom.idx, o).order == 2 for o in oxygens)
                if atom.charge == 1 or has_double_o:
                    groups.append(PerceivedGroup("nitro", False, adj_atoms[0], {atom.idx} | set(oxygens)))
                    consumed.update([atom.idx] + oxygens)
            elif len(oxygens) == 1 and len(adj_atoms) == 1:
                if mol.get_bond(atom.idx, oxygens[0]).order == 2:
                    groups.append(PerceivedGroup("nitroso", False, adj_atoms[0], {atom.idx, oxygens[0]}))
                    consumed.update([atom.idx, oxygens[0]])

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            nitrogens =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N"]
            adj_atoms =[n for n in mol.get_neighbors(atom.idx) if n not in nitrogens]
            if len(adj_atoms) == 1 and len(nitrogens) == 1:
                n2 = nitrogens[0]
                n2_nitrogens =[n for n in mol.get_neighbors(n2) if mol.atoms[n].symbol == "N" and n != atom.idx]
                if len(n2_nitrogens) == 1:
                    n3 = n2_nitrogens[0]
                    if mol.degree(n3) == 1:
                        groups.append(PerceivedGroup("azido", False, adj_atoms[0], {atom.idx, n2, n3}))
                        consumed.update([atom.idx, n2, n3])

    for atom in mol:
        if atom.symbol == "O" and mol.degree(atom.idx) == 2 and atom.idx not in consumed:
            adj_atoms = mol.get_neighbors(atom.idx)
            if len(adj_atoms) == 2:
                c1, c2 = adj_atoms
                if mol.atoms[c1].is_carbon and mol.atoms[c2].is_carbon:
                    o1 = next((o for o in mol.get_neighbors(c1) if mol.atoms[o].symbol == "O" and mol.get_bond(c1, o).order == 2), None)
                    o2 = next((o for o in mol.get_neighbors(c2) if mol.atoms[o].symbol == "O" and mol.get_bond(c2, o).order == 2), None)
                    if o1 and o2:
                        visited = {atom.idx, c1}
                        q =[c1]
                        is_cyclic = False
                        while q:
                            curr = q.pop(0)
                            for nxt in mol.get_neighbors(curr):
                                if nxt == c2:
                                    is_cyclic = True
                                    break
                                if nxt not in visited:
                                    visited.add(nxt)
                                    q.append(nxt)
                            if is_cyclic: break
                            
                        if not is_cyclic:
                            groups.append(PerceivedGroup("anhydride", True, c1, {atom.idx, o1, o2}))
                            consumed.update([atom.idx, o1, o2])

    for atom in mol:
        if atom.symbol == "S" and atom.idx not in consumed:
            oxygens =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "O"]
            adj_atoms =[n for n in mol.get_neighbors(atom.idx) if n not in oxygens]
            if len(oxygens) >= 3 and len(adj_atoms) == 1:
                double_o_list =[o for o in oxygens if mol.get_bond(atom.idx, o).order == 2]
                if len(double_o_list) >= 2:
                    c_idx = adj_atoms[0]
                    single_o_list =[o for o in oxygens if mol.get_bond(atom.idx, o).order == 1]
                    ester_o = next((o for o in single_o_list if mol.degree(o) == 2), None)
                    anion_o = next((o for o in single_o_list if mol.atoms[o].charge == -1), None)
                    
                    if ester_o is not None or anion_o is not None:
                        key = "sulfonate"
                        groups.append(PerceivedGroup(key, True, c_idx, {atom.idx} | set(oxygens)))
                        consumed.update([atom.idx] + oxygens)
                    elif len(single_o_list) > 0:
                        key = "sulfonic_acid"
                        groups.append(PerceivedGroup(key, True, c_idx, {atom.idx} | set(oxygens)))
                        consumed.update([atom.idx] + oxygens)

    for atom in mol:
        if atom.is_carbon and atom.idx not in consumed:
            nitrogens =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N" and n not in consumed]
            oxygens =[o for o in mol.get_neighbors(atom.idx) if mol.atoms[o].symbol == "O" and o not in consumed]
            sulfurs =[s for s in mol.get_neighbors(atom.idx) if mol.atoms[s].symbol == "S" and s not in consumed]
            
            double_n = next((n for n in nitrogens if mol.get_bond(atom.idx, n).order == 2), None)
            double_o = next((o for o in oxygens if mol.get_bond(atom.idx, o).order == 2 and mol.degree(o) == 1), None)
            double_s = next((s for s in sulfurs if mol.get_bond(atom.idx, s).order == 2 and mol.degree(s) == 1), None)
            
            if double_n is not None:
                if double_s is not None:
                    n_neighbors =[x for x in mol.get_neighbors(double_n) if x != atom.idx]
                    if len(n_neighbors) > 0:
                        groups.append(PerceivedGroup("isothiocyanato", False, n_neighbors[0], {atom.idx, double_n, double_s}))
                        consumed.update([atom.idx, double_n, double_s])
                        continue
                elif double_o is not None:
                    n_neighbors =[x for x in mol.get_neighbors(double_n) if x != atom.idx]
                    if len(n_neighbors) > 0:
                        groups.append(PerceivedGroup("isocyanato", False, n_neighbors[0], {atom.idx, double_n, double_o}))
                        consumed.update([atom.idx, double_n, double_o])
                        continue
                        
            triple_n = next((n for n in nitrogens if mol.get_bond(atom.idx, n).order == 3), None)
            single_o = next((o for o in oxygens if mol.get_bond(atom.idx, o).order == 1), None)
            single_s = next((s for s in sulfurs if mol.get_bond(atom.idx, s).order == 1), None)
            
            if triple_n is not None:
                if single_s is not None:
                    s_neighbors =[x for x in mol.get_neighbors(single_s) if x != atom.idx]
                    if len(s_neighbors) > 0:
                        groups.append(PerceivedGroup("thiocyanato", False, s_neighbors[0], {atom.idx, triple_n, single_s}))
                        consumed.update([atom.idx, triple_n, single_s])
                        continue
                elif single_o is not None:
                    o_neighbors =[x for x in mol.get_neighbors(single_o) if x != atom.idx]
                    if len(o_neighbors) > 0:
                        groups.append(PerceivedGroup("cyanato", False, o_neighbors[0], {atom.idx, triple_n, single_o}))
                        consumed.update([atom.idx, triple_n, single_o])
                        continue

    for atom in mol:
        if atom.is_carbon and atom.idx not in consumed:
            nitrogens =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N" and n not in consumed]
            triple_n = next((n for n in nitrogens if mol.get_bond(atom.idx, n).order == 3), None)
            if triple_n is not None:
                n_neighbors =[x for x in mol.get_neighbors(triple_n) if x != atom.idx]
                if len(n_neighbors) > 0:
                    groups.append(PerceivedGroup("isocyano", False, n_neighbors[0], {atom.idx, triple_n}))
                    consumed.update([atom.idx, triple_n])
                else:
                    ring_neighbors =[n for n in mol.get_neighbors(atom.idx) if n in cyclic_atoms]
                    is_exocyclic = False
                    attached_ring_atom = None
                    if atom.idx not in cyclic_atoms and len(ring_neighbors) == 1:
                        attached_ring_atom = ring_neighbors[0]
                        if mol.get_bond(atom.idx, attached_ring_atom).order == 1:
                            is_exocyclic = True
                    target_carbon = attached_ring_atom if is_exocyclic else atom.idx
                    key = "ring_nitrile" if is_exocyclic else "nitrile"
                    groups.append(PerceivedGroup(key, True, target_carbon, {atom.idx, triple_n}))
                    consumed.update([triple_n])

    for atom in mol:
        if atom.is_carbon:
            oxygens =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "O" and n not in consumed]
            nitrogens =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N" and n not in consumed]
            halogens =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol in["F", "Cl", "Br", "I"] and n not in consumed]
            
            double_o = next((o for o in oxygens if mol.get_bond(atom.idx, o).order == 2 and mol.degree(o) == 1), None)
            single_o = next((o for o in oxygens if mol.get_bond(atom.idx, o).order == 1), None)
            single_n = next((n for n in nitrogens if mol.get_bond(atom.idx, n).order == 1), None)
            single_x = next((x for x in halogens if mol.get_bond(atom.idx, x).order == 1), None)
            
            if double_o is not None:
                ring_neighbors =[n for n in mol.get_neighbors(atom.idx) if n in cyclic_atoms]
                is_exocyclic = False
                attached_ring_atom = None
                
                if atom.idx not in cyclic_atoms and len(ring_neighbors) == 1:
                    attached_ring_atom = ring_neighbors[0]
                    if mol.get_bond(atom.idx, attached_ring_atom).order == 1:
                        is_exocyclic = True
                        
                target_carbon = attached_ring_atom if is_exocyclic else atom.idx
                
                if single_o is not None:
                    o_neighbors =[x for x in mol.get_neighbors(single_o) if x != atom.idx]
                    is_peroxy = False
                    peroxy_o = None
                    if len(o_neighbors) == 1 and mol.atoms[o_neighbors[0]].symbol == "O":
                        is_peroxy = True
                        peroxy_o = o_neighbors[0]
                    elif len(o_neighbors) > 1 and any(mol.atoms[x].symbol == "O" for x in o_neighbors):
                        is_peroxy = True
                        peroxy_o = next(x for x in o_neighbors if mol.atoms[x].symbol == "O")
                        
                    if is_peroxy:
                        if mol.degree(peroxy_o) == 1 or mol.atoms[peroxy_o].charge == -1:
                            key = "ring_peroxy_acid" if is_exocyclic else "peroxy_acid"
                        else:
                            key = "ring_peroxy_ester" if is_exocyclic else "peroxy_ester"
                        groups.append(PerceivedGroup(key, True, target_carbon, {atom.idx, double_o, single_o, peroxy_o}))
                        consumed.update([double_o, single_o, peroxy_o])
                    else:
                        is_lactone = False
                        if target_carbon == atom.idx and single_o in cyclic_atoms:
                            visited = {atom.idx, double_o}
                            q = [single_o]
                            while q:
                                curr = q.pop(0)
                                for nxt in mol.get_neighbors(curr):
                                    if nxt == atom.idx and curr != single_o:
                                        is_lactone = True
                                        break
                                    if nxt not in visited and nxt in cyclic_atoms:
                                        visited.add(nxt)
                                        q.append(nxt)
                                if is_lactone: break

                        if is_lactone:
                            groups.append(PerceivedGroup("ketone", True, target_carbon, {atom.idx, double_o}))
                            consumed.update([double_o]) 
                        else:
                            if mol.atoms[single_o].charge == -1:
                                key = "ring_carboxylate" if is_exocyclic else "carboxylate"
                            elif mol.degree(single_o) == 1:
                                key = "ring_carboxylic_acid" if is_exocyclic else "carboxylic_acid"
                            else:
                                key = "ring_carboxylate" if is_exocyclic else "ester"
                            groups.append(PerceivedGroup(key, True, target_carbon, {atom.idx, double_o, single_o}))
                            consumed.update([double_o, single_o])
                elif single_n is not None:
                    is_lactam = False
                    if target_carbon == atom.idx and single_n in cyclic_atoms:
                        visited = {atom.idx, double_o}
                        q = [single_n]
                        while q:
                            curr = q.pop(0)
                            for nxt in mol.get_neighbors(curr):
                                if nxt == atom.idx and curr != single_n:
                                    is_lactam = True
                                    break
                                if nxt not in visited and nxt in cyclic_atoms:
                                    visited.add(nxt)
                                    q.append(nxt)
                            if is_lactam: break

                    if is_lactam:
                        groups.append(PerceivedGroup("ketone", True, target_carbon, {atom.idx, double_o}))
                        consumed.update([double_o])
                    else:
                        if single_n in cyclic_atoms:
                            pass
                        else:
                            key = "ring_amide" if is_exocyclic else "amide"
                            groups.append(PerceivedGroup(key, True, target_carbon, {atom.idx, double_o, single_n}))
                            consumed.update([double_o, single_n])
                elif single_x is not None:
                    sym = mol.atoms[single_x].symbol
                    if is_exocyclic:
                        x_map = {"F": "ring_acid_fluoride", "Cl": "ring_acid_chloride", "Br": "ring_acid_bromide", "I": "ring_acid_iodide"}
                    else:
                        x_map = {"F": "acid_fluoride", "Cl": "acid_chloride", "Br": "acid_bromide", "I": "acid_iodide"}
                    groups.append(PerceivedGroup(x_map[sym], True, target_carbon, {atom.idx, double_o, single_x}))
                    consumed.update([double_o, single_x])

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            c_neighbors =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].is_carbon]
            n_neighbors =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N"]
            
            double_c = next((c for c in c_neighbors if mol.get_bond(atom.idx, c).order == 2), None)
            
            if double_c is not None:
                if len(n_neighbors) > 0:
                    n2 = n_neighbors[0]
                    if mol.get_bond(atom.idx, n2).order == 1:
                        if n2 not in cyclic_atoms:
                            ring_neighbors =[n for n in mol.get_neighbors(double_c) if n in cyclic_atoms]
                            c_of_double_c =[n for n in mol.get_neighbors(double_c) if mol.atoms[n].is_carbon]
                            if double_c not in cyclic_atoms and len(ring_neighbors) == 1 and len(c_of_double_c) == 1 and mol.get_bond(double_c, ring_neighbors[0]).order == 1:
                                groups.append(PerceivedGroup("ring_aldehyde_hydrazone", True, ring_neighbors[0], {double_c, atom.idx, n2}))
                            else:
                                if len(c_of_double_c) <= 1 and double_c not in cyclic_atoms:
                                    groups.append(PerceivedGroup("aldehyde_hydrazone", True, double_c, {atom.idx, n2}))
                                else:
                                    groups.append(PerceivedGroup("hydrazone", True, double_c, {atom.idx, n2}))
                            consumed.update([atom.idx, n2])
                        else:
                            groups.append(PerceivedGroup("imine", True, double_c, {atom.idx}))
                            consumed.update([atom.idx])
                    else:
                        groups.append(PerceivedGroup("imine", True, double_c, {atom.idx}))
                        consumed.update([atom.idx])
                else:
                    groups.append(PerceivedGroup("imine", True, double_c, {atom.idx}))
                    consumed.update([atom.idx])

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            c_neighbors =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].is_carbon]
            n_neighbors =[n for n in mol.get_neighbors(atom.idx) if mol.atoms[n].symbol == "N"]
            
            if len(n_neighbors) > 0:
                n2 = n_neighbors[0]
                c_att = c_neighbors[0] if c_neighbors else None
                if c_att:
                    if mol.get_bond(atom.idx, n2).order == 1:
                        if n2 not in cyclic_atoms:
                            groups.append(PerceivedGroup("hydrazine", True, c_att, {atom.idx, n2}))
                            consumed.update([atom.idx, n2])
                        else:
                            groups.append(PerceivedGroup("amine", True, c_att, {atom.idx}))
                            consumed.update([atom.idx])

    for atom in mol:
        if atom.idx not in consumed:
            if atom.symbol == "O" and mol.degree(atom.idx) == 1:
                adj_atoms = mol.get_neighbors(atom.idx)
                if len(adj_atoms) == 1:
                    c_idx = adj_atoms[0]
                    bond = mol.get_bond(atom.idx, c_idx)
                    if bond.order == 2:
                        if mol.atoms[c_idx].is_carbon:
                            if c_idx in cyclic_atoms:
                                groups.append(PerceivedGroup("ketone", True, c_idx, {atom.idx}))
                            elif len(mol.get_neighbors(c_idx)) >= 3:
                                groups.append(PerceivedGroup("ketone", True, c_idx, {atom.idx}))
                            else:
                                ring_neighbors =[n for n in mol.get_neighbors(c_idx) if n in cyclic_atoms]
                                if len(ring_neighbors) == 1 and mol.get_bond(c_idx, ring_neighbors[0]).order == 1:
                                    groups.append(PerceivedGroup("ring_aldehyde", True, ring_neighbors[0], {c_idx, atom.idx}))
                                else:
                                    groups.append(PerceivedGroup("aldehyde", True, c_idx, {c_idx, atom.idx}))
                    elif bond.order == 1:
                        if mol.atoms[c_idx].is_carbon:
                            groups.append(PerceivedGroup("alcohol", True, c_idx, {atom.idx}))
                    consumed.add(atom.idx)
            elif atom.symbol == "S" and mol.degree(atom.idx) == 1 and atom.idx not in cyclic_atoms:
                adj_atoms = mol.get_neighbors(atom.idx)
                if len(adj_atoms) == 1:
                    c_idx = adj_atoms[0]
                    bond = mol.get_bond(atom.idx, c_idx)
                    if bond.order == 1:
                        if mol.atoms[c_idx].is_carbon:
                            groups.append(PerceivedGroup("thiol", True, c_idx, {atom.idx}))
                            consumed.add(atom.idx)

    for atom in mol:
        if atom.symbol == "N" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            adj_atoms = mol.get_neighbors(atom.idx)
            if len(adj_atoms) > 0:
                for c in adj_atoms:
                    groups.append(PerceivedGroup("amine", True, c, {atom.idx}))
                consumed.add(atom.idx)

    for atom in mol:
        if atom.symbol == "O" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            adj_atoms = mol.get_neighbors(atom.idx)
            if len(adj_atoms) == 2:
                for c in adj_atoms:
                    groups.append(PerceivedGroup("ether", False, c, {atom.idx}))
                consumed.add(atom.idx)

    for atom in mol:
        if atom.symbol == "S" and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            adj_atoms = mol.get_neighbors(atom.idx)
            if len(adj_atoms) == 2:
                for c in adj_atoms:
                    groups.append(PerceivedGroup("thioether", False, c, {atom.idx}))
                consumed.add(atom.idx)

    halogen_map = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}
    for atom in mol:
        if atom.symbol in halogen_map and atom.idx not in consumed and atom.idx not in cyclic_atoms:
            if mol.degree(atom.idx) == 1:
                adj_atoms = mol.get_neighbors(atom.idx)
                if len(adj_atoms) == 1:
                    groups.append(PerceivedGroup(halogen_map[atom.symbol], False, adj_atoms[0], {atom.idx}))
                    consumed.add(atom.idx)

    return groups
