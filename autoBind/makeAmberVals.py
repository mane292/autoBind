#!/usr/bin/env python
import parmed as pmd
from metallicious import supramolecular_structure


###INPUTS####
input_pdb = "PPP_p2Me.pdb"
metal_chgs = {'Pd':2 }

cage = supramolecular_structure(input_pdb, metal_charges=metal_chgs, LJ_type='merz-opc')
cage.prepare_initial_topology()
cage.parametrize(out_coord=f'cage_out.pdb', out_topol=f'cage_out.top')

# Load GROMACS topology and structure
print("Loading GROMACS topology...")
gro_top = pmd.load_file('cage_out.top', xyz='cage_out.pdb')

# Save as Amber format
print("Saving Amber topology and coordinates...")
gro_top.save('cage_out.prmtop', format='amber')
gro_top.save('cage_out.inpcrd', format='rst7')

print("Conversion complete!")
print("Created: cage_out.prmtop and cage_out.inpcrd")
