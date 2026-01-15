#!/usr/bin/env python3
"""
prmtop_to_exact_frcmod.py

Generate:
  (1) MOL2 from Amber prmtop + inpcrd
  (2) "Exact-match" frcmod extracted from the prmtop: all parameters that involve
      at least one "special" atom type.

Requires: parmed (AmberTools)
Usage:
  python prmtop_to_exact_frcmod.py out.prmtop out.inpcrd out.mol2 out_exact.frcmod ":LIG"
or (metal auto mode):
  python prmtop_to_exact_frcmod.py out.prmtop out.inpcrd out.mol2 out_exact.frcmod "METAL_AUTO:2.6"
"""

import sys
import math
import parmed as pmd
from parmed.amber import AmberMask

def collect_special_types(struct, selector: str):
    selector = selector.strip()

    # METAL_AUTO unchanged (if you still want it); keep your existing block here if you like
    if selector.upper().startswith("METAL_AUTO:"):
        cutoff = float(selector.split(":", 1)[1])
        # ... your existing METAL_AUTO logic ...
        # return special_types, special_atoms
        raise RuntimeError("METAL_AUTO block not shown here; keep your existing one above if needed.")

    # Always use AmberMask for residue/atom masks
    mask = AmberMask(struct, selector)
    indices = mask.Selected()  # list of atom indices

    if not indices:
        raise RuntimeError(f"Mask '{selector}' selected 0 atoms. Check residue names.")

    special_atoms = {struct.atoms[i] for i in indices}
    special_types = {a.type for a in special_atoms if a.type is not None}

    if not special_types:
        raise RuntimeError(f"Mask '{selector}' selected atoms but none had atom types.")

    return special_types, special_atoms

def fmt(x, w=12, p=6):
    return f"{x:{w}.{p}f}"

def is_metal_element(elem: str) -> bool:
    # simple heuristic; customize if needed
    metals = {
        "LI","NA","K","RB","CS","MG","CA","SR","BA","ZN","CU","FE","CO","NI","MN",
        "AG","AU","CD","HG","AL","GA","IN","SN","PB","SB","BI","TI","V","CR","MO",
        "W","PD","PT","RU","RH","IR","OS"
    }
    return (elem or "").strip().upper() in metals

def dist(a, b):
    dx = a.xx - b.xx
    dy = a.xy - b.xy
    dz = a.xz - b.xz
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def main():
    # ===== CONFIGURABLE DEFAULTS =====
    # You can modify these values as needed
    default_prmtop = "cage_out.prmtop"
    default_inpcrd = "cage_out.incprd"
    default_out_mol2 = "CAGE.mol2"
    default_out_frcmod = "CAGE.frcmod"
    default_selector = ":*"  # Select all residues/atoms
    # =================================

    # Use command line arguments if provided, otherwise use defaults
    if len(sys.argv) >= 6:
        prmtop, inpcrd, out_mol2, out_frcmod, selector = sys.argv[1:6]
    else:
        prmtop = default_prmtop
        inpcrd = default_inpcrd
        out_mol2 = default_out_mol2
        out_frcmod = default_out_frcmod
        selector = default_selector
        print(f"Using default parameters:")
        print(f"  Input prmtop:  {prmtop}")
        print(f"  Input inpcrd:  {inpcrd}")
        print(f"  Output mol2:   {out_mol2}")
        print(f"  Output frcmod: {out_frcmod}")
        print(f"  Selector:      {selector}")
        print()
    # --- write MOL2 (entire system) ---
    # Note: MOL2 for a huge solvated system will be huge; if you want only a subset,
    # do that via mask in a quick tweak below.
    struct = pmd.load_file(prmtop, inpcrd)
    struct.load_atom_info()
    print("Before load_atom_info:", [a.type for a in struct.atoms[:10]])

    # --- identify special types ---
    special_types, special_atoms = collect_special_types(struct, selector)
    print(f"Selector {selector!r}: selected {len(special_atoms)} atoms, {len(special_types)} types")
    sel_idx = sorted(a.idx for a in special_atoms)
    sub = struct[sel_idx]                       # subset structure
    sub.save(out_mol2, format="mol2", overwrite=True)
    out_pdb = out_mol2.rsplit(".", 1)[0] + ".pdb"
    sub.save(out_pdb, format="pdb", overwrite=True)

    struct = pmd.load_file(prmtop, inpcrd)
    if not special_types:
        raise RuntimeError("No special types identified (types missing?).")

    def involves_special(types):
        return any(t in special_types for t in types)

    # --- MASS section (for special types) ---
    # Use the first observed atom of each type to define mass.
    type_to_mass = {}
    for a in struct.atoms:
        if a.type in special_types and a.type not in type_to_mass:
            type_to_mass[a.type] = a.mass

    # --- BOND / ANGLE / DIHE / IMPROPER ---
    bonds = {}   # key(sorted types) -> (k, req)
    angles = {}  # key(types in order) -> (k, theta_deg)
    diheds = {}  # key(types) -> set of terms (pk, phase_deg, per)
    improps = {} # same format

    # Bonds
    for b in struct.bonds:
        t1, t2 = b.atom1.type, b.atom2.type
        if t1 is None or t2 is None or b.type is None:
            continue
        if involves_special([t1, t2]):
            key = tuple(sorted((t1, t2)))
            bonds[key] = (b.type.k, b.type.req)

    # Angles
    for a in struct.angles:
        t1, t2, t3 = a.atom1.type, a.atom2.type, a.atom3.type
        if None in (t1, t2, t3) or a.type is None:
            continue
        if involves_special([t1, t2, t3]):
            key = (t1, t2, t3)
            angles[key] = (a.type.k, a.type.theteq)

    # Dihedrals (proper + improper)
    for d in struct.dihedrals:
        t1, t2, t3, t4 = d.atom1.type, d.atom2.type, d.atom3.type, d.atom4.type
        if None in (t1, t2, t3, t4) or d.type is None:
            continue
        if not involves_special([t1, t2, t3, t4]):
            continue
        term = (d.type.phi_k, d.type.phase, d.type.per)
        key = (t1, t2, t3, t4)
        if d.improper:
            improps.setdefault(key, set()).add(term)
        else:
            diheds.setdefault(key, set()).add(term)

    # --- NONBON (LJ) ---
    # Use ParmEd AtomType sigma/epsilon when available.
    nonbon = {}  # type -> (sigma, epsilon)
    for a in struct.atoms:
        t = a.type
        if t in special_types and a.atom_type is not None:
            sigma = getattr(a.atom_type, "sigma", None)
            eps = getattr(a.atom_type, "epsilon", None)
            if sigma is None or eps is None:
                # fallback: rmin (sigma = rmin*2^(-1/6))
                rmin = getattr(a.atom_type, "rmin", None)
                eps2 = getattr(a.atom_type, "epsilon", None)
                if rmin is None or eps2 is None:
                    continue
                sigma = rmin * (2.0 ** (-1.0/6.0))
                eps = eps2
            nonbon[t] = (sigma, eps)

    # --- Write frcmod ---
    with open(out_frcmod, "w") as f:
        f.write("EXACT_FROM_PRMTOP\n\n")

        f.write("MASS\n")
        for t in sorted(type_to_mass.keys()):
            f.write(f"{t:<6} {fmt(type_to_mass[t], w=10, p=4)}\n")

        f.write("\nBOND\n")
        for (t1, t2), (k, req) in sorted(bonds.items()):
            f.write(f"{t1:<2}-{t2:<2}  {fmt(k)}  {fmt(req)}\n")

        f.write("\nANGLE\n")
        for (t1, t2, t3), (k, th) in sorted(angles.items()):
            f.write(f"{t1:<2}-{t2:<2}-{t3:<2}  {fmt(k)}  {fmt(th)}\n")

        f.write("\nDIHE\n")
        for (t1, t2, t3, t4), terms in sorted(diheds.items()):
            for (pk, phase, per) in sorted(terms):
                f.write(f"{t1:<2}-{t2:<2}-{t3:<2}-{t4:<2}  {fmt(pk)}  {fmt(phase)}  {fmt(per)}\n")

        f.write("\nIMPROPER\n")
        for (t1, t2, t3, t4), terms in sorted(improps.items()):
            for (pk, phase, per) in sorted(terms):
                f.write(f"{t1:<2}-{t2:<2}-{t3:<2}-{t4:<2}  {fmt(pk)}  {fmt(phase)}  {fmt(per)}\n")

        f.write("\nNONBON\n")
        for t, (sigma, eps) in sorted(nonbon.items()):
            f.write(f"{t:<6} {fmt(sigma)}  {fmt(eps)}\n")

        f.write("\nEND\n")

    print("Wrote:", out_mol2)
    print("Wrote:", out_frcmod)
    print("Special selector:", selector)
    print("Special types:", " ".join(sorted(special_types)))

if __name__ == "__main__":
    main()
