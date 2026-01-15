"""
autoBind: Automated AMBER topology generation for cage structures.

This package provides tools to automatically generate AMBER topology files
for metal-organic cage structures with counterions and substrates.

Quick Start
-----------
>>> from autobind import AutoBind, run_autobind
>>>
>>> # Simple usage with chemical names (recommended)
>>> ab = AutoBind(
...     input_pdb="my_cage.pdb",
...     counterion_type='BArF',        # Tetrakis(3,5-bis(trifluoromethyl)phenyl)borate
...     substrate='pToluquinone',       # para-Toluquinone
...     solvent='DCM'                   # Dichloromethane
... )
>>> ab.run_all()
>>>
>>> # Or use the convenience function
>>> ab = run_autobind("my_cage.pdb")

Available Components
--------------------
Counterion types: BArF
Substrates: pToluquinone, PTQ, toluquinone
Solvents: DCM, dichloromethane, CH2Cl2

To see all available components:
>>> AutoBind.list_available()
"""

__author__ = """Melissa T. Manetsch"""
__email__ = 'manets12@mit.edu'
__version__ = "0.1.0"

from .autobind import AutoBind, run_autobind, Atom, Mol2Atom

__all__ = ['AutoBind', 'run_autobind', 'Atom', 'Mol2Atom']
