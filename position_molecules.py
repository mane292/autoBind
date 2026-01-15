#!/usr/bin/env python3
"""
Script to position counterions outside the cage and substrate inside the cage,
then update all relevant files.

CONFIGURATION: Edit the variables in the CONFIGURATION section below to control behavior.
"""

import numpy as np
import re
from typing import List, Dict, Tuple
import copy

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES TO CONTROL BEHAVIOR
# ============================================================================
#PPP2NO2_large.pdb 

# Input/Output files
INPUT_PDB_FILE = 'pre_PPP_p2Me_mcpbpy.pdb'  # Input PDB file with cage structure
OUTPUT_PDB_FILE = 'PPP_p2Me_mcpbpy.pdb'  # Output PDB file (can be same as input to overwrite)

# What to process
PROCESS_COUNTERIONS = True  # Set to True to position counterions
PROCESS_SUBSTRATE = True    # Set to True to position substrate

# Counterion settings
COUNTERION_NAMES = ['BFY', 'BFW', 'BFX', 'BFV']  # List of counterion residue names
COUNTERION_RESIDS = [7, 8, 9, 10]  # Residue IDs for counterions in output
COUNTERION_PLACEMENT_MULTIPLIER = 2.0  # Distance multiplier from cage center (1.5-2.5 typical)

# Substrate settings
SUBSTRATE_NAME = 'PZQ'  # Substrate residue name
SUBSTRATE_RESID = 11    # Residue ID for substrate in output
SUBSTRATE_COVALENT_CUTOFF = 2.5  # Minimum distance to linkers (Angstroms)

# ============================================================================
# END CONFIGURATION
# ============================================================================

class Atom:
    """Represents an atom with its properties."""
    def __init__(self, serial, name, resname, resid, x, y, z, occupancy=1.0, tempfactor=0.0):
        self.serial = serial
        self.name = name
        self.resname = resname
        self.resid = resid
        self.x = x
        self.y = y
        self.z = z
        self.occupancy = occupancy
        self.tempfactor = tempfactor

    def coords(self):
        return np.array([self.x, self.y, self.z])

    def set_coords(self, coords):
        self.x, self.y, self.z = coords

    def to_pdb_line(self):
        """Convert atom to PDB format line."""
        return f"HETATM{self.serial:5d}  {self.name:<3s} {self.resname:3s} A{self.resid:4d}    {self.x:8.3f}{self.y:8.3f}{self.z:8.3f}{self.occupancy:6.2f}{self.tempfactor:6.2f}\n"


class Mol2Atom:
    """Represents an atom from mol2 file."""
    def __init__(self, atom_id, name, x, y, z, atom_type, resid, resname, charge):
        self.atom_id = atom_id
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.atom_type = atom_type
        self.resid = resid
        self.resname = resname
        self.charge = charge

    def coords(self):
        return np.array([self.x, self.y, self.z])

    def set_coords(self, coords):
        self.x, self.y, self.z = coords

    def to_mol2_line(self):
        """Convert atom to mol2 format line."""
        return f"{self.atom_id:7d} {self.name:<8s} {self.x:11.4f} {self.y:11.4f} {self.z:11.4f} {self.atom_type:<10s} {self.resid:>2d} {self.resname:<8s} {self.charge:9.6f}\n"


def read_pdb(filename):
    """Read PDB file and return atoms grouped by residue."""
    residues = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                serial = int(line[6:11])
                name = line[12:16].strip()
                resname = line[17:20].strip()
                resid = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occupancy = float(line[54:60]) if len(line) > 54 else 1.0
                tempfactor = float(line[60:66]) if len(line) > 60 else 0.0

                atom = Atom(serial, name, resname, resid, x, y, z, occupancy, tempfactor)

                if resid not in residues:
                    residues[resid] = []
                residues[resid].append(atom)

    return residues


def read_mol2(filename):
    """Read mol2 file and return atoms and bonds."""
    atoms = []
    bonds = []
    molecule_info = {}

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find sections
    in_molecule = False
    in_atom = False
    in_bond = False

    for i, line in enumerate(lines):
        if '@<TRIPOS>MOLECULE' in line:
            in_molecule = True
            in_atom = False
            in_bond = False
            # Read molecule name
            molecule_info['name'] = lines[i+1].strip()
            continue
        elif '@<TRIPOS>ATOM' in line:
            in_atom = True
            in_molecule = False
            in_bond = False
            continue
        elif '@<TRIPOS>BOND' in line:
            in_bond = True
            in_atom = False
            in_molecule = False
            continue
        elif '@<TRIPOS>SUBSTRUCTURE' in line:
            in_bond = False
            break

        if in_atom and line.strip():
            parts = line.split()
            if len(parts) >= 9:
                atom_id = int(parts[0])
                name = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                atom_type = parts[5]
                resid = int(parts[6])
                resname = parts[7]
                charge = float(parts[8])

                atom = Mol2Atom(atom_id, name, x, y, z, atom_type, resid, resname, charge)
                atoms.append(atom)

        elif in_bond and line.strip():
            bonds.append(line)

    return atoms, bonds, molecule_info, lines


def write_mol2(filename, atoms, bonds, molecule_info, original_lines):
    """Write mol2 file with updated atoms."""
    with open(filename, 'w') as f:
        # Write header sections
        in_atom = False
        in_bond = False

        for line in original_lines:
            if '@<TRIPOS>ATOM' in line:
                in_atom = True
                in_bond = False
                f.write(line)
                # Write all atoms
                for atom in atoms:
                    f.write(atom.to_mol2_line())
                continue
            elif '@<TRIPOS>BOND' in line:
                in_bond = True
                in_atom = False
                f.write(line)
                # Write all bonds
                for bond in bonds:
                    f.write(bond)
                continue
            elif '@<TRIPOS>SUBSTRUCTURE' in line:
                in_bond = False
                in_atom = False

            if not in_atom and not in_bond:
                f.write(line)


def calculate_cage_bounds(residues):
    """Calculate the bounds of the cage structure."""
    # Get Pd positions (residues 1 and 2)
    pd1_coords = residues[1][0].coords()
    pd2_coords = residues[2][0].coords()

    # Get all linker atoms (residues 3-6)
    linker_coords = []
    for resid in [3, 4, 5, 6]:
        for atom in residues[resid]:
            linker_coords.append(atom.coords())

    linker_coords = np.array(linker_coords)

    # Calculate cage center (midpoint between Pd ions)
    cage_center = (pd1_coords + pd2_coords) / 2

    # Calculate cage bounds
    all_cage_coords = np.vstack([linker_coords, [pd1_coords, pd2_coords]])
    min_bounds = np.min(all_cage_coords, axis=0)
    max_bounds = np.max(all_cage_coords, axis=0)

    # Calculate max distance from center to any cage atom
    distances = np.linalg.norm(all_cage_coords - cage_center, axis=1)
    max_radius = np.max(distances)

    return cage_center, min_bounds, max_bounds, max_radius, pd1_coords, pd2_coords, linker_coords


def position_counterions_outside(cage_center, max_radius, counterion_atoms_list, placement_multiplier=2.0):
    """Position counterions outside the cage in a tetrahedral arrangement."""
    # Tetrahedral vertices relative to center
    # These are normalized vectors pointing to the vertices of a tetrahedron
    tetrahedron_directions = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ])

    # Normalize
    tetrahedron_directions = tetrahedron_directions / np.linalg.norm(tetrahedron_directions, axis=1)[:, np.newaxis]

    # Place counterions outside the cage
    placement_distance = max_radius * placement_multiplier

    positioned_counterions = []

    for i, atoms in enumerate(counterion_atoms_list):
        # Calculate center of mass of counterion
        coords = np.array([atom.coords() for atom in atoms])
        com = np.mean(coords, axis=0)

        # Calculate new center position
        new_center = cage_center + tetrahedron_directions[i] * placement_distance

        # Translate all atoms
        translation = new_center - com
        new_atoms = []
        for atom in atoms:
            new_atom = copy.deepcopy(atom)
            new_atom.set_coords(atom.coords() + translation)
            new_atoms.append(new_atom)

        positioned_counterions.append(new_atoms)

    return positioned_counterions


def position_substrate_inside(cage_center, pd1_coords, pd2_coords, linker_coords, substrate_atoms, covalent_cutoff=2.5):
    """Position substrate inside the cage, avoiding linkers and counterions."""
    # Calculate substrate center of mass
    coords = np.array([atom.coords() for atom in substrate_atoms])
    substrate_com = np.mean(coords, axis=0)

    # Calculate the vector from PD1 to PD2
    pd_vector = pd2_coords - pd1_coords
    pd_vector_norm = pd_vector / np.linalg.norm(pd_vector)

    # Try multiple positions along and around the PD1-PD2 axis
    best_position = cage_center.copy()
    best_min_distance = 0

    # Try positions along the PD axis with finer grid
    for offset_along in np.linspace(-3, 3, 13):
        for theta in np.linspace(0, 2*np.pi, 12):
            for r in np.linspace(0, 2.5, 6):
                # Position along PD axis with circular offset
                perp_vector1 = np.array([pd_vector_norm[1], -pd_vector_norm[0], 0])
                if np.linalg.norm(perp_vector1) < 0.1:
                    perp_vector1 = np.array([0, pd_vector_norm[2], -pd_vector_norm[1]])
                perp_vector1 = perp_vector1 / np.linalg.norm(perp_vector1)
                perp_vector2 = np.cross(pd_vector_norm, perp_vector1)

                circular_offset = r * (np.cos(theta) * perp_vector1 + np.sin(theta) * perp_vector2)
                target_position = cage_center + offset_along * pd_vector_norm + circular_offset

                # Calculate translation
                translation = target_position - substrate_com

                # Get translated substrate coordinates
                translated_coords = coords + translation

                # Check distances to linker atoms
                min_linker_distance = float('inf')
                for linker_coord in linker_coords:
                    distances = np.linalg.norm(translated_coords - linker_coord, axis=1)
                    min_dist = np.min(distances)
                    if min_dist < min_linker_distance:
                        min_linker_distance = min_dist

                # Keep track of best position
                if min_linker_distance > best_min_distance:
                    best_min_distance = min_linker_distance
                    best_position = target_position.copy()

                # If we found a good position, use it
                if min_linker_distance > covalent_cutoff:
                    best_position = target_position
                    break
            if best_min_distance > covalent_cutoff:
                break
        if best_min_distance > covalent_cutoff:
            break

    # Apply final translation
    translation = best_position - substrate_com
    positioned_substrate = []
    for atom in substrate_atoms:
        new_atom = copy.deepcopy(atom)
        new_atom.set_coords(atom.coords() + translation)
        positioned_substrate.append(new_atom)

    print(f"   Best position found with min distance to linkers: {best_min_distance:.3f} Å")

    return positioned_substrate


def update_residue_numbers(atoms, new_resid, new_resname=None):
    """Update residue numbers in a list of atoms."""
    for atom in atoms:
        atom.resid = new_resid
        if new_resname:
            atom.resname = new_resname


def mol2_to_pdb_atoms(mol2_atoms, resid, resname, starting_serial):
    """Convert mol2 atoms to PDB atoms."""
    pdb_atoms = []
    for i, mol2_atom in enumerate(mol2_atoms):
        pdb_atom = Atom(
            serial=starting_serial + i,
            name=mol2_atom.name,
            resname=resname,
            resid=resid,
            x=mol2_atom.x,
            y=mol2_atom.y,
            z=mol2_atom.z,
            occupancy=1.0,
            tempfactor=0.0
        )
        pdb_atoms.append(pdb_atom)
    return pdb_atoms


def write_pdb(filename, residues_dict):
    """Write PDB file from residues dictionary with sequential atom numbering."""
    with open(filename, 'w') as f:
        f.write("REMARK, BUILD BY MCPB.PY\n")

        # Renumber atoms sequentially to ensure chronological ordering
        atom_serial = 1
        for resid in sorted(residues_dict.keys()):
            atoms = residues_dict[resid]
            for atom in atoms:
                # Temporarily update serial number for output
                original_serial = atom.serial
                atom.serial = atom_serial
                f.write(atom.to_pdb_line())
                atom.serial = original_serial  # Restore original (in case needed elsewhere)
                atom_serial += 1
            f.write("TER\n")

        f.write("END\n")


def process_counterions(cage_center, max_radius, counterion_names=['BFY', 'BFW', 'BFX', 'BFV'],
                        new_resids=[7, 8, 9, 10], placement_multiplier=2.0):
    """
    Position counterions outside the cage.

    Args:
        cage_center: Center of the cage
        max_radius: Maximum radius of the cage
        counterion_names: List of counterion residue names
        new_resids: List of new residue IDs for counterions
        placement_multiplier: Distance multiplier from cage center

    Returns:
        Dictionary containing counterion data with positioned atoms
    """
    print("\n" + "="*70)
    print("COUNTERION POSITIONING MODULE")
    print("="*70)

    # Read counterion mol2 files
    print("\n1. Reading counterion mol2 files...")
    counterion_data = {}

    for name in counterion_names:
        atoms, bonds, mol_info, orig_lines = read_mol2(f'{name}.mol2')
        counterion_data[name] = {
            'atoms': atoms,
            'bonds': bonds,
            'mol_info': mol_info,
            'orig_lines': orig_lines
        }
        print(f"   {name}: {len(atoms)} atoms")

    # Position counterions outside cage
    print(f"\n2. Positioning counterions outside cage (distance = {placement_multiplier}x radius)...")
    counterion_atoms_list = [counterion_data[name]['atoms'] for name in counterion_names]
    positioned_counterions = position_counterions_outside(cage_center, max_radius, counterion_atoms_list, placement_multiplier)

    # Update residue numbers for counterions
    print("\n3. Updating counterion residue numbers...")
    for i, (name, new_resid) in enumerate(zip(counterion_names, new_resids)):
        update_residue_numbers(positioned_counterions[i], new_resid, name)
        counterion_data[name]['atoms'] = positioned_counterions[i]
        print(f"   {name} → Residue {new_resid}")

    # Write updated counterion mol2 files
    print("\n4. Writing updated counterion mol2 files...")
    for name in counterion_names:
        write_mol2(
            f'{name}.mol2',
            counterion_data[name]['atoms'],
            counterion_data[name]['bonds'],
            counterion_data[name]['mol_info'],
            counterion_data[name]['orig_lines']
        )
        print(f"   Updated {name}.mol2")

    print("\n✓ Counterion positioning complete!")
    return counterion_data, new_resids


def process_substrate(cage_center, pd1_coords, pd2_coords, linker_coords,
                      substrate_name='PZQ', substrate_resid=11, covalent_cutoff=2.5):
    """
    Position substrate inside the cage.

    Args:
        cage_center: Center of the cage
        pd1_coords: Coordinates of first Pd ion
        pd2_coords: Coordinates of second Pd ion
        linker_coords: Array of linker atom coordinates
        substrate_name: Name of substrate residue
        substrate_resid: Residue number for substrate
        covalent_cutoff: Minimum distance to linkers (Angstroms)

    Returns:
        Tuple of (positioned atoms, bonds, mol_info, orig_lines)
    """
    print("\n" + "="*70)
    print("SUBSTRATE POSITIONING MODULE")
    print("="*70)

    # Read substrate mol2 file
    print(f"\n1. Reading {substrate_name} substrate mol2 file...")
    substrate_atoms, substrate_bonds, substrate_mol_info, substrate_orig_lines = read_mol2(f'{substrate_name}.mol2')
    print(f"   {substrate_name}: {len(substrate_atoms)} atoms")

    # Position substrate inside cage
    print(f"\n2. Positioning {substrate_name} substrate inside cage...")
    positioned_substrate = position_substrate_inside(
        cage_center, pd1_coords, pd2_coords, linker_coords, substrate_atoms, covalent_cutoff=covalent_cutoff
    )

    # Update residue number for substrate
    print(f"\n3. Updating {substrate_name} residue number to {substrate_resid}...")
    update_residue_numbers(positioned_substrate, substrate_resid, substrate_name)

    # Write updated substrate mol2 file
    print(f"\n4. Writing updated {substrate_name} mol2 file...")
    write_mol2(f'{substrate_name}.mol2', positioned_substrate, substrate_bonds, substrate_mol_info, substrate_orig_lines)
    print(f"   Updated {substrate_name}.mol2")

    print(f"\n✓ Substrate positioning complete!")
    return positioned_substrate, substrate_bonds, substrate_mol_info, substrate_orig_lines


def build_complete_structure(pdb_residues, counterion_data=None, counterion_resids=None,
                             substrate_atoms=None, substrate_name='PZQ', substrate_resid=11):
    """
    Build complete PDB structure with cage, counterions, and/or substrate.

    Args:
        pdb_residues: Original cage residues
        counterion_data: Dictionary of counterion data (optional)
        counterion_resids: List of counterion residue IDs (optional)
        substrate_atoms: Positioned substrate atoms (optional)
        substrate_name: Name of substrate
        substrate_resid: Residue ID for substrate

    Returns:
        Complete residues dictionary
    """
    print("\n" + "="*70)
    print("BUILDING COMPLETE STRUCTURE")
    print("="*70)

    complete_pdb_residues = copy.deepcopy(pdb_residues)

    # Get the last atom serial number
    last_serial = max([atom.serial for residue in pdb_residues.values() for atom in residue])

    # Add counterions if provided
    if counterion_data is not None and counterion_resids is not None:
        print("\nAdding counterions to structure...")
        counterion_names = list(counterion_data.keys())
        for i, (name, new_resid) in enumerate(zip(counterion_names, counterion_resids)):
            pdb_atoms = mol2_to_pdb_atoms(
                counterion_data[name]['atoms'],
                new_resid,
                name,
                last_serial + 1
            )
            complete_pdb_residues[new_resid] = pdb_atoms
            last_serial += len(pdb_atoms)
            print(f"  Added {name} as residue {new_resid}")

    # Add substrate if provided
    if substrate_atoms is not None:
        print(f"\nAdding {substrate_name} to structure...")
        substrate_pdb_atoms = mol2_to_pdb_atoms(substrate_atoms, substrate_resid, substrate_name, last_serial + 1)
        complete_pdb_residues[substrate_resid] = substrate_pdb_atoms
        print(f"  Added {substrate_name} as residue {substrate_resid}")

    return complete_pdb_residues


def print_summary(complete_pdb_residues, has_counterions=True, has_substrate=True):
    """Print summary of the final structure."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total residues in final structure: {len(complete_pdb_residues)}")
    print(f"  1: PD1 (Pd ion)")
    print(f"  2: PD2 (Pd ion)")
    print(f"  3: LA1 (linker)")
    print(f"  4: LB1 (linker)")
    print(f"  5: LC1 (linker)")
    print(f"  6: LD1 (linker)")

    if has_counterions:
        print(f"  7: BFY (counterion, outside cage)")
        print(f"  8: BFW (counterion, outside cage)")
        print(f"  9: BFX (counterion, outside cage)")
        print(f" 10: BFV (counterion, outside cage)")

    if has_substrate:
        print(f" 11: PZQ (substrate, inside cage)")

    print("\nAll files updated successfully!")
    print("=" * 70)


def main():
    """Main function using configuration variables defined at top of file."""
    print("=" * 70)
    if PROCESS_COUNTERIONS and PROCESS_SUBSTRATE:
        print("Positioning Counterions and Substrate for Cage System")
    elif PROCESS_COUNTERIONS:
        print("Positioning Counterions for Cage System")
    else:
        print("Positioning Substrate for Cage System")
    print("=" * 70)

    # Read the original PDB file
    print(f"\nReading {INPUT_PDB_FILE}...")
    pdb_residues = read_pdb(INPUT_PDB_FILE)
    print(f"  Found {len(pdb_residues)} residues")

    # Calculate cage bounds
    print("\nAnalyzing cage structure...")
    cage_center, min_bounds, max_bounds, max_radius, pd1_coords, pd2_coords, linker_coords = calculate_cage_bounds(pdb_residues)
    print(f"  Cage center: ({cage_center[0]:.3f}, {cage_center[1]:.3f}, {cage_center[2]:.3f})")
    print(f"  Cage radius: {max_radius:.3f} Å")
    print(f"  PD1 at: ({pd1_coords[0]:.3f}, {pd1_coords[1]:.3f}, {pd1_coords[2]:.3f})")
    print(f"  PD2 at: ({pd2_coords[0]:.3f}, {pd2_coords[1]:.3f}, {pd2_coords[2]:.3f})")

    # Process counterions
    counterion_data = None
    counterion_resids = None
    if PROCESS_COUNTERIONS:
        counterion_data, counterion_resids = process_counterions(
            cage_center, max_radius,
            counterion_names=COUNTERION_NAMES,
            new_resids=COUNTERION_RESIDS,
            placement_multiplier=COUNTERION_PLACEMENT_MULTIPLIER
        )

    # Process substrate
    substrate_atoms = None
    if PROCESS_SUBSTRATE:
        substrate_atoms, _, _, _ = process_substrate(
            cage_center, pd1_coords, pd2_coords, linker_coords,
            substrate_name=SUBSTRATE_NAME,
            substrate_resid=SUBSTRATE_RESID,
            covalent_cutoff=SUBSTRATE_COVALENT_CUTOFF
        )

    # Build complete structure
    complete_pdb_residues = build_complete_structure(
        pdb_residues,
        counterion_data=counterion_data,
        counterion_resids=counterion_resids,
        substrate_atoms=substrate_atoms,
        substrate_name=SUBSTRATE_NAME,
        substrate_resid=SUBSTRATE_RESID
    )

    # Write complete PDB file
    print(f"\nWriting complete PDB file to {OUTPUT_PDB_FILE}...")
    write_pdb(OUTPUT_PDB_FILE, complete_pdb_residues)
    print(f"  Updated {OUTPUT_PDB_FILE}")

    # Print summary
    print_summary(complete_pdb_residues,
                 has_counterions=PROCESS_COUNTERIONS,
                 has_substrate=PROCESS_SUBSTRATE)


if __name__ == '__main__':
    main()
