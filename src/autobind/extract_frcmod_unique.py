"""
Extract frcmod with unique atom types to preserve all parameters.

This module creates unique atom types for atoms that have different parameter
environments, ensuring that metallicious-generated parameters are preserved
when going through tleap.
"""

import parmed as pmd
from parmed.amber import AmberMask
from collections import defaultdict
from pathlib import Path
import math


def extract_frcmod_unique_types(
    prmtop: str,
    inpcrd: str,
    out_mol2: str = "CAGE.mol2",
    out_frcmod: str = "CAGE.frcmod",
    selector: str = ":*",
    verbose: bool = True
):
    """
    Extract force field parameters from prmtop to frcmod format,
    creating unique atom types to preserve all parameters.

    Parameters
    ----------
    prmtop : str
        Path to prmtop file
    inpcrd : str
        Path to inpcrd file
    out_mol2 : str
        Output mol2 file name
    out_frcmod : str
        Output frcmod file name
    selector : str
        Atom selection mask (default ":*" for all atoms)
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Mapping of atom index to new unique type
    """

    if verbose:
        print(f"\nLoading {prmtop}...")
    struct = pmd.load_file(prmtop, inpcrd)

    # Get selected atom indices
    mask = AmberMask(struct, selector)
    sel_idx = sorted(list(mask.Selected()))
    sel_idx_set = set(sel_idx)

    if verbose:
        print(f"Selected {len(sel_idx)} atoms")

    # Step 1: Build parameter signature for each atom
    # Signature = tuple of all bond/angle/dihedral parameters the atom participates in

    atom_signatures = {i: [] for i in sel_idx}

    # Collect bond parameters
    for b in struct.bonds:
        if b.type is None:
            continue
        i1, i2 = b.atom1.idx, b.atom2.idx
        if i1 not in sel_idx_set or i2 not in sel_idx_set:
            continue
        params = (round(b.type.k, 6), round(b.type.req, 6))
        atom_signatures[i1].append(('B', b.atom2.type, params))
        atom_signatures[i2].append(('B', b.atom1.type, params))

    # Collect angle parameters
    for a in struct.angles:
        if a.type is None:
            continue
        i1, i2, i3 = a.atom1.idx, a.atom2.idx, a.atom3.idx
        if i1 not in sel_idx_set or i2 not in sel_idx_set or i3 not in sel_idx_set:
            continue
        params = (round(a.type.k, 6), round(a.type.theteq, 6))
        atom_signatures[i1].append(('A', a.atom2.type, a.atom3.type, params))
        atom_signatures[i2].append(('A', a.atom1.type, a.atom3.type, params))
        atom_signatures[i3].append(('A', a.atom1.type, a.atom2.type, params))

    # Step 2: Group atoms by (original_type, signature) and assign unique types
    type_sig_to_atoms = defaultdict(list)
    for idx in sel_idx:
        orig_type = struct.atoms[idx].type
        sig = tuple(sorted(atom_signatures[idx]))
        type_sig_to_atoms[(orig_type, sig)].append(idx)

    # Step 3: Create unique type names
    # Group by original type first to see which need splitting
    orig_type_groups = defaultdict(list)
    for (orig_type, sig), indices in type_sig_to_atoms.items():
        orig_type_groups[orig_type].append((sig, indices))

    atom_to_new_type = {}
    new_type_to_mass = {}
    type_counter = {}

    if verbose:
        print("\nCreating unique atom types:")

    for orig_type, sig_groups in sorted(orig_type_groups.items()):
        if len(sig_groups) == 1:
            # Only one signature - keep original type (truncated to 2 chars for AMBER)
            new_type = orig_type[:2]
            sig, indices = sig_groups[0]
            for idx in indices:
                atom_to_new_type[idx] = new_type
            new_type_to_mass[new_type] = struct.atoms[indices[0]].mass
            if verbose:
                print(f"  {orig_type} -> {new_type} ({len(indices)} atoms)")
        else:
            # Multiple signatures - create unique types
            # Use base letter + suffix format, keeping to 2 chars total for AMBER compatibility
            # Suffix: 0-9 for first 10, then A-Z for next 26 (total 36 unique per base)
            base = orig_type[0].lower()
            if base not in type_counter:
                type_counter[base] = 0

            for sig, indices in sig_groups:
                count = type_counter[base]
                if count < 10:
                    suffix = str(count)
                elif count < 36:
                    suffix = chr(ord('A') + count - 10)
                else:
                    # Fallback for >36 types: use lowercase (might conflict but rare)
                    suffix = chr(ord('a') + count - 36)
                new_type = f"{base}{suffix}"
                type_counter[base] += 1
                for idx in indices:
                    atom_to_new_type[idx] = new_type
                new_type_to_mass[new_type] = struct.atoms[indices[0]].mass

            if verbose:
                print(f"  {orig_type} -> {len(sig_groups)} unique types ({sum(len(g[1]) for g in sig_groups)} atoms)")

    # Step 4: Collect all parameters using NEW types
    bonds = {}
    angles = {}
    diheds = defaultdict(set)
    improps = defaultdict(set)

    for b in struct.bonds:
        if b.type is None:
            continue
        i1, i2 = b.atom1.idx, b.atom2.idx
        if i1 not in sel_idx_set or i2 not in sel_idx_set:
            continue
        t1 = atom_to_new_type[i1]
        t2 = atom_to_new_type[i2]
        key = tuple(sorted((t1, t2)))
        bonds[key] = (b.type.k, b.type.req)

    for a in struct.angles:
        if a.type is None:
            continue
        i1, i2, i3 = a.atom1.idx, a.atom2.idx, a.atom3.idx
        if i1 not in sel_idx_set or i2 not in sel_idx_set or i3 not in sel_idx_set:
            continue
        t1 = atom_to_new_type[i1]
        t2 = atom_to_new_type[i2]
        t3 = atom_to_new_type[i3]
        key = (t1, t2, t3)
        angles[key] = (a.type.k, a.type.theteq)

    for d in struct.dihedrals:
        if d.type is None:
            continue
        i1, i2, i3, i4 = d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx
        if not all(i in sel_idx_set for i in [i1, i2, i3, i4]):
            continue
        t1 = atom_to_new_type[i1]
        t2 = atom_to_new_type[i2]
        t3 = atom_to_new_type[i3]
        t4 = atom_to_new_type[i4]
        term = (d.type.phi_k, d.type.phase, d.type.per)
        key = (t1, t2, t3, t4)
        if d.improper:
            improps[key].add(term)
        else:
            diheds[key].add(term)

    # Nonbonded parameters
    # AMBER frcmod NONBON section expects R* = Rmin/2 (half the minimum energy distance)
    # parmed's rmin attribute is actually Rmin, so we need to divide by 2
    nonbon = {}
    for idx in sel_idx:
        atom = struct.atoms[idx]
        new_type = atom_to_new_type[idx]
        if new_type in nonbon:
            continue
        if atom.atom_type is not None:
            # Try to get rmin first (preferred for AMBER)
            rmin = getattr(atom.atom_type, "rmin", None)
            eps = getattr(atom.atom_type, "epsilon", None)
            if rmin is None:
                # Fall back to sigma and convert to rmin
                sigma = getattr(atom.atom_type, "sigma", None)
                if sigma is not None:
                    rmin = sigma * (2.0 ** (1.0/6.0))
            if rmin is not None and eps is not None:
                # AMBER frcmod expects R* = Rmin/2
                r_star = rmin / 2.0
                nonbon[new_type] = (r_star, eps)

    # Step 5: Write frcmod file
    if verbose:
        print(f"\nWriting {out_frcmod}...")
        print(f"  {len(bonds)} bond types")
        print(f"  {len(angles)} angle types")
        print(f"  {len(diheds)} dihedral types")
        print(f"  {len(improps)} improper types")
        print(f"  {len(nonbon)} nonbonded types")

    def fmt(x, w=12, p=6):
        return f"{x:{w}.{p}f}"

    with open(out_frcmod, "w") as f:
        f.write("Auto-generated frcmod with unique atom types\n")
        f.write("MASS\n")
        for t in sorted(new_type_to_mass.keys()):
            f.write(f"{t:<6} {fmt(new_type_to_mass[t], w=10, p=4)}\n")

        f.write("\nBOND\n")
        for (t1, t2), (k, req) in sorted(bonds.items()):
            f.write(f"{t1:<2}-{t2:<2}  {fmt(k)}  {fmt(req)}\n")

        f.write("\nANGLE\n")
        for (t1, t2, t3), (k, th) in sorted(angles.items()):
            # parmed already returns theteq in degrees
            f.write(f"{t1:<2}-{t2:<2}-{t3:<2}  {fmt(k)}  {fmt(th)}\n")

        f.write("\nDIHE\n")
        for (t1, t2, t3, t4), terms in sorted(diheds.items()):
            for (pk, phase, per) in sorted(terms):
                # parmed already returns phase in degrees
                # AMBER frcmod format: AT1-AT2-AT3-AT4  IDIVF  PK  PHASE  PN
                f.write(f"{t1:<2}-{t2:<2}-{t3:<2}-{t4:<2}   1  {fmt(pk)}  {fmt(phase)}  {fmt(per)}\n")

        f.write("\nIMPROPER\n")
        for (t1, t2, t3, t4), terms in sorted(improps.items()):
            for (pk, phase, per) in sorted(terms):
                # parmed already returns phase in degrees
                f.write(f"{t1:<2}-{t2:<2}-{t3:<2}-{t4:<2}  {fmt(pk)}  {fmt(phase)}  {fmt(per)}\n")

        f.write("\nNONBON\n")
        for t, (r_star, eps) in sorted(nonbon.items()):
            f.write(f"{t:<6} {fmt(r_star)}  {fmt(eps)}\n")

        f.write("\nEND\n")

    # Step 6: Write mol2 file with new atom types
    if verbose:
        print(f"\nWriting {out_mol2}...")

    # Create subset structure
    if len(sel_idx) == len(struct.atoms):
        sub = struct[:]
    else:
        sub = struct[sel_idx]

    # Update atom types in the subset
    for i, atom in enumerate(sub.atoms):
        orig_idx = sel_idx[i]
        atom.type = atom_to_new_type[orig_idx]

    sub.save(out_mol2, format="mol2", overwrite=True)

    # Also save PDB
    out_pdb = out_mol2.rsplit(".", 1)[0] + ".pdb"
    sub.save(out_pdb, format="pdb", overwrite=True)

    if verbose:
        print(f"\nDone! Created {len(set(atom_to_new_type.values()))} unique atom types.")

    return atom_to_new_type


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_frcmod_unique.py <prmtop> <inpcrd> [out_mol2] [out_frcmod]")
        sys.exit(1)

    prmtop = sys.argv[1]
    inpcrd = sys.argv[2]
    out_mol2 = sys.argv[3] if len(sys.argv) > 3 else "CAGE.mol2"
    out_frcmod = sys.argv[4] if len(sys.argv) > 4 else "CAGE.frcmod"

    extract_frcmod_unique_types(prmtop, inpcrd, out_mol2, out_frcmod)
