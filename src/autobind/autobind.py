#!/usr/bin/env python3
"""
AutoBind: Automated AMBER topology generation for cage structures.

This module combines functionality from:
- makeAmberVals.py: Generate AMBER topology from metallicious
- extract_frcmod.py: Extract force field parameters from prmtop
- position_molecules.py: Position counterions and substrates
- tleap generation and execution

Example usage:
    from autobind import AutoBind

    ab = AutoBind(
        input_pdb="my_cage.pdb",
        metal_charges={'Pd': 2},
        counterions=['BFV', 'BFW', 'BFX', 'BFY'],
        substrate='PZQ',
        solvent='DCM'
    )
    ab.run_all()
"""

import os
import sys
import math
import copy
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np

try:
    import parmed as pmd
    from parmed.amber import AmberMask
except ImportError:
    pmd = None
    AmberMask = None

try:
    from metallicious import supramolecular_structure
except ImportError:
    supramolecular_structure = None

try:
    from .extract_frcmod_unique import extract_frcmod_unique_types
except ImportError:
    extract_frcmod_unique_types = None


# Get the data directory path (inside the package)
DATA_DIR = Path(__file__).parent / "data"


class Atom:
    """Represents an atom with its properties for PDB files."""
    def __init__(self, serial: int, name: str, resname: str, resid: int,
                 x: float, y: float, z: float, occupancy: float = 1.0, tempfactor: float = 0.0):
        self.serial = serial
        self.name = name
        self.resname = resname
        self.resid = resid
        self.x = x
        self.y = y
        self.z = z
        self.occupancy = occupancy
        self.tempfactor = tempfactor

    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def set_coords(self, coords: np.ndarray):
        self.x, self.y, self.z = coords

    def to_pdb_line(self) -> str:
        """Convert atom to PDB format line."""
        return f"HETATM{self.serial:5d}  {self.name:<3s} {self.resname:3s} A{self.resid:4d}    {self.x:8.3f}{self.y:8.3f}{self.z:8.3f}{self.occupancy:6.2f}{self.tempfactor:6.2f}\n"


class Mol2Atom:
    """Represents an atom from mol2 file."""
    def __init__(self, atom_id: int, name: str, x: float, y: float, z: float,
                 atom_type: str, resid: int, resname: str, charge: float):
        self.atom_id = atom_id
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.atom_type = atom_type
        self.resid = resid
        self.resname = resname
        self.charge = charge

    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def set_coords(self, coords: np.ndarray):
        self.x, self.y, self.z = coords

    def to_mol2_line(self) -> str:
        """Convert atom to mol2 format line."""
        return f"{self.atom_id:7d} {self.name:<8s} {self.x:11.4f} {self.y:11.4f} {self.z:11.4f} {self.atom_type:<10s} {self.resid:>2d} {self.resname:<8s} {self.charge:9.6f}\n"


class AutoBind:
    """
    Main class for automated AMBER topology generation for cage structures.

    Combines the functionality of makeAmberVals.py, extract_frcmod.py, and
    position_molecules.py into a single class with configurable inputs.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file containing the cage structure.
    metal_charges : dict, optional
        Dictionary mapping metal symbols to their charges. Default is {'Pd': 2}.
    counterion_type : str, optional
        Type of counterion to use. Use chemical names like 'BArF' which will
        automatically expand to the appropriate residue files. Default is 'BArF'.
        Available types: 'BArF' (tetrakis(3,5-bis(trifluoromethyl)phenyl)borate)
    substrate : str, optional
        Name of the substrate to use. Use chemical names like 'pToluquinone'.
        Default is 'pToluquinone'. Available: 'pToluquinone', 'PZQ'
    solvent : str, optional
        Name of the solvent to use. Default is 'DCM' (dichloromethane).
        Available: 'DCM'
    parameter_set : str, optional
        Path to the parameter set file. Default is DCMgaff2.dat in data folder.
    working_dir : str, optional
        Working directory for output files. Default is current directory.
    process_counterions : bool, optional
        Whether to process counterions. Default is True.
    process_substrate : bool, optional
        Whether to process substrate. Default is True.
    counterion_placement_multiplier : float, optional
        Distance multiplier from cage center for counterion placement. Default is 2.0.
    substrate_covalent_cutoff : float, optional
        Minimum distance to linkers for substrate placement (Angstroms). Default is 2.5.
    lj_type : str, optional
        LJ type for metallicious. Default is 'merz-opc'.
    debug : bool, optional
        Enable verbose debug output. Default is False.

    Examples
    --------
    >>> # Simple usage with chemical names
    >>> ab = AutoBind(
    ...     input_pdb="my_cage.pdb",
    ...     counterion_type='BArF',      # Automatically uses BFV, BFW, BFX, BFY
    ...     substrate='pToluquinone'      # Automatically maps to PZQ files
    ... )
    >>> ab.run_all()

    >>> # Without substrate
    >>> ab = AutoBind(
    ...     input_pdb="my_cage.pdb",
    ...     counterion_type='BArF',
    ...     process_substrate=False
    ... )

    >>> # With debug output enabled
    >>> ab = AutoBind(
    ...     input_pdb="my_cage.pdb",
    ...     debug=True
    ... )
    """

    # =========================================================================
    # CHEMICAL LIBRARIES - Maps common names to internal file identifiers
    # =========================================================================

    # Counterion type library: maps chemical names to list of residue names
    # Each counterion type can have multiple residues (e.g., 4 BArF anions)
    # For built-in counterions, 'builtin' flag indicates using tleap's addIons command
    COUNTERION_LIBRARY = {
        'BArF': {
            'full_name': 'Tetrakis(3,5-bis(trifluoromethyl)phenyl)borate',
            'residues': ['BFV', 'BFW', 'BFX', 'BFY'],
            'charge': -1,
            'description': 'Weakly coordinating anion, 4 copies for Pd2L4 cage'
        },
        # Built-in counterions (use tleap's addIons command, no custom files needed)
        'Cl': {
            'full_name': 'Chloride',
            'builtin': True,
            'ion_name': 'Cl-',
            'charge': -1,
            'count': 4,  # Number of counterions to add (for +4 cage charge)
            'description': 'Chloride anion using tleap built-in parameters'
        },
        'chloride': {'builtin': True, 'ion_name': 'Cl-', 'charge': -1, 'count': 4},  # Alias
        'Cl-': {'builtin': True, 'ion_name': 'Cl-', 'charge': -1, 'count': 4},  # Alias
    }

    # Substrate library: maps chemical names to internal residue names
    SUBSTRATE_LIBRARY = {
        'pToluquinone': {
            'full_name': 'para-Toluquinone (2-methyl-1,4-benzoquinone)',
            'residue': 'PZQ',
            'description': 'Quinone substrate for cage binding studies'
        },
        'p-toluquinone': {'residue': 'PZQ'},  # Alias
        'PTQ': {'residue': 'PZQ'},  # Alias
        'toluquinone': {'residue': 'PZQ'},  # Alias
    }

    # Solvent library: maps common names to internal identifiers
    SOLVENT_LIBRARY = {
        'DCM': {
            'full_name': 'Dichloromethane',
            'identifier': 'DCM',
            'description': 'Non-polar aprotic solvent'
        },
        'dichloromethane': {'identifier': 'DCM'},  # Alias
        'CH2Cl2': {'identifier': 'DCM'},  # Alias
        'ACE': {
            'full_name': 'Acetone',
            'identifier': 'ACE',
            'description': 'Polar aprotic solvent'
        },
        'acetone': {'identifier': 'ACE'},  # Alias
        'DMSO': {
            'full_name': 'Dimethyl sulfoxide',
            'identifier': 'DMSO',
            'description': 'Polar aprotic solvent'
        },
        'dimethyl sulfoxide': {'identifier': 'DMSO'},  # Alias
        'HCN': {
            'full_name': 'Acetonitrile',
            'identifier': 'HCN',
            'description': 'Polar aprotic solvent'
        },
        'acetonitrile': {'identifier': 'HCN'},  # Alias
        'ACN': {'identifier': 'HCN'},  # Alias
        'MeCN': {'identifier': 'HCN'},  # Alias
        'Nitro': {
            'full_name': 'Nitromethane',
            'identifier': 'Nitro',
            'description': 'Polar aprotic solvent'
        },
        'nitromethane': {'identifier': 'Nitro'},  # Alias
        'MeNO2': {'identifier': 'Nitro'},  # Alias
        'THF': {
            'full_name': 'Tetrahydrofuran',
            'identifier': 'THF',
            'description': 'Polar aprotic solvent'
        },
        'tetrahydrofuran': {'identifier': 'THF'},  # Alias
        'oDFB': {
            'full_name': 'ortho-Difluorobenzene',
            'identifier': 'oDFB',
            'description': 'Fluorinated aromatic solvent'
        },
        'orthodifluorobenzene': {'identifier': 'oDFB'},  # Alias
        'o-difluorobenzene': {'identifier': 'oDFB'},  # Alias
        '1,2-difluorobenzene': {'identifier': 'oDFB'},  # Alias
        # Water models (use built-in tleap water boxes)
        'water': {
            'full_name': 'Water (OPC model)',
            'identifier': 'OPC',
            'description': 'OPC water model (default water)'
        },
        'OPC': {
            'full_name': 'OPC Water',
            'identifier': 'OPC',
            'description': 'OPC 4-point water model'
        },
        'opc': {'identifier': 'OPC'},  # Alias
        'TIP3P': {
            'full_name': 'TIP3P Water',
            'identifier': 'TIP3P',
            'description': 'TIP3P 3-point water model'
        },
        'tip3p': {'identifier': 'TIP3P'},  # Alias
    }

    # =========================================================================
    # FILE MAPPINGS - Maps residue names to actual files in data folders
    # =========================================================================

    # Available counterion residues and their files (in data/counterions/)
    AVAILABLE_COUNTERIONS = {
        'BFV': {'mol2': 'BFV.mol2', 'frcmod': 'BFV.frcmod'},
        'BFW': {'mol2': 'BFW.mol2', 'frcmod': 'BFW.frcmod'},
        'BFX': {'mol2': 'BFX.mol2', 'frcmod': 'BFX.frcmod'},
        'BFY': {'mol2': 'BFY.mol2', 'frcmod': 'BFY.frcmod'},
    }

    # Available substrate residues and their files (in data/binding_substrates/)
    AVAILABLE_SUBSTRATES = {
        'PZQ': {'mol2': 'PZQ.mol2', 'frcmod': 'PZQ.frcmod', 'pdb': 'PZQ.pdb'},
    }

    # Available solvents and their files (in data/solvent_box_info/)
    # For water models, 'builtin' flag indicates using tleap's built-in water box
    AVAILABLE_SOLVENTS = {
        'DCM': {'lib': 'DCMequilbox.lib', 'box_name': 'DCMequilbox'},
        'ACE': {'lib': 'ACEequilbox.lib', 'box_name': 'ACEequilbox'},
        'DMSO': {'lib': 'DMSOequilbox.lib', 'box_name': 'DMSOequilbox'},
        'HCN': {'lib': 'HCNequilbox.lib', 'box_name': 'HCNequilbox'},
        'Nitro': {'lib': 'Nitroequilbox.lib', 'box_name': 'Nitroequilbox'},
        'THF': {'lib': 'THFequilbox.lib', 'box_name': 'THFequilbox'},
        'oDFB': {'lib': 'oDFBequilbox.lib', 'box_name': 'oDFBequilbox'},
        # Built-in water models (no lib file needed)
        'OPC': {'builtin': True, 'box_name': 'OPCBOX', 'leaprc': 'leaprc.water.opc'},
        'TIP3P': {'builtin': True, 'box_name': 'TIP3PBOX', 'leaprc': 'leaprc.water.tip3p'},
    }

    def _debug_print(self, message: str, level: int = 0):
        """Print debug message if debug mode is enabled.

        Parameters
        ----------
        message : str
            The debug message to print.
        level : int, optional
            Indentation level (each level adds 2 spaces). Default is 0.
        """
        if getattr(self, 'debug', False):
            indent = "  " * level
            print(f"[DEBUG] {indent}{message}")

    def __init__(
        self,
        input_pdb: str,
        metal_charges: Optional[Dict[str, int]] = None,
        counterion_type: str = 'BArF',
        substrate: str = 'pToluquinone',
        solvent: str = 'DCM',
        parameter_set: Optional[str] = None,
        working_dir: Optional[str] = None,
        process_counterions: bool = True,
        process_substrate: bool = True,
        counterion_placement_multiplier: float = 2.0,
        substrate_covalent_cutoff: float = 2.5,
        lj_type: str = 'merz-opc',
        debug: bool = False
    ):
        self.debug = debug
        self._debug_print("="*70)
        self._debug_print("AUTOBIND DEBUG: Initializing AutoBind instance")
        self._debug_print("="*70)
        self._debug_print(f"  input_pdb: {input_pdb}")
        self._debug_print(f"  metal_charges: {metal_charges}")
        self._debug_print(f"  counterion_type: {counterion_type}")
        self._debug_print(f"  substrate: {substrate}")
        self._debug_print(f"  solvent: {solvent}")
        self._debug_print(f"  working_dir: {working_dir}")
        self._debug_print(f"  process_counterions: {process_counterions}")
        self._debug_print(f"  process_substrate: {process_substrate}")

        self.input_pdb = input_pdb
        self.metal_charges = metal_charges or {'Pd': 2}
        self.counterion_type = counterion_type
        self.substrate_name = substrate  # Store the user-friendly name
        self.solvent_name = solvent  # Store the user-friendly name
        self.process_counterions = process_counterions
        self.process_substrate = process_substrate
        self.counterion_placement_multiplier = counterion_placement_multiplier
        self.substrate_covalent_cutoff = substrate_covalent_cutoff
        self.lj_type = lj_type

        # Initialize built-in flags (will be set by resolve methods if needed)
        self._builtin_counterion = False
        self._builtin_counterion_info = None

        # Resolve chemical names to internal identifiers
        self.counterions = self._resolve_counterion_type(counterion_type)
        self.substrate = self._resolve_substrate_name(substrate)
        self.solvent = self._resolve_solvent_name(solvent)

        # Set working directory
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Set parameter set path - default to data folder location
        if parameter_set:
            self.parameter_set = Path(parameter_set)
        else:
            # Try data folder first, then parent autoBind folder
            default_param = DATA_DIR.parent.parent / "DCMgaff2.dat"
            if default_param.exists():
                self.parameter_set = default_param
            else:
                self.parameter_set = Path("/home/gridsan/mmanetsch/DCMgaff2.dat")

        # Data directories
        self.counterions_dir = DATA_DIR / "counterions"
        self.substrates_dir = DATA_DIR / "binding_substrates"
        self.solvent_dir = DATA_DIR / "solvent_box_info"

        # Validate inputs
        self._debug_print("Validating inputs...")
        self._validate_inputs()
        self._debug_print("Input validation complete")

        # Generated file paths (will be set during processing)
        self._debug_print("Setting up file path variables...")
        self.cage_pdb = None
        self.cage_top = None
        self.cage_prmtop = None
        self.cage_inpcrd = None
        self.cage_mol2 = None
        self.cage_frcmod = None
        self.final_pdb = None

        # Residue IDs for counterions and substrate
        self._counterion_resids = list(range(7, 7 + len(self.counterions)))
        self._substrate_resid = 7 + len(self.counterions) if self.process_substrate else None

    def _resolve_counterion_type(self, counterion_type: str) -> List[str]:
        """
        Resolve a counterion type name to a list of residue names.

        Parameters
        ----------
        counterion_type : str
            Chemical name of the counterion type (e.g., 'BArF').

        Returns
        -------
        List[str]
            List of residue names corresponding to this counterion type.
            Empty list for built-in counterions (handled by tleap addIons).
        """
        if not self.process_counterions:
            return []

        # Check if it's a known counterion type
        if counterion_type in self.COUNTERION_LIBRARY:
            ci_info = self.COUNTERION_LIBRARY[counterion_type]

            # Check if it's a built-in counterion (uses tleap's addIons)
            if ci_info.get('builtin', False):
                self._builtin_counterion = True
                self._builtin_counterion_info = ci_info
                print(f"Counterion type '{counterion_type}' -> built-in (tleap addIons {ci_info.get('ion_name', 'Cl-')})")
                return []  # No custom residues needed

            residues = ci_info['residues']
            print(f"Counterion type '{counterion_type}' -> {residues}")
            return residues

        # Check if it's directly a residue name (for backwards compatibility)
        if counterion_type in self.AVAILABLE_COUNTERIONS:
            return [counterion_type]

        # Not found
        available_types = list(self.COUNTERION_LIBRARY.keys())
        raise ValueError(
            f"Counterion type '{counterion_type}' not found.\n"
            f"Available counterion types: {available_types}\n"
            f"Or use individual residue names: {list(self.AVAILABLE_COUNTERIONS.keys())}"
        )

    def _resolve_substrate_name(self, substrate: str) -> Optional[str]:
        """
        Resolve a substrate chemical name to its internal residue name.

        Parameters
        ----------
        substrate : str
            Chemical name of the substrate (e.g., 'pToluquinone').

        Returns
        -------
        str or None
            Internal residue name (e.g., 'PZQ'), or None if no substrate.
        """
        if not self.process_substrate or substrate is None:
            return None

        # Check if it's in the substrate library
        if substrate in self.SUBSTRATE_LIBRARY:
            residue = self.SUBSTRATE_LIBRARY[substrate]['residue']
            full_name = self.SUBSTRATE_LIBRARY[substrate].get('full_name', substrate)
            print(f"Substrate '{substrate}' ({full_name}) -> {residue}")
            return residue

        # Check if it's directly a residue name (for backwards compatibility)
        if substrate in self.AVAILABLE_SUBSTRATES:
            return substrate

        # Not found
        available_names = [k for k, v in self.SUBSTRATE_LIBRARY.items() if 'full_name' in v]
        raise ValueError(
            f"Substrate '{substrate}' not found.\n"
            f"Available substrate names: {available_names}\n"
            f"Or use internal residue names: {list(self.AVAILABLE_SUBSTRATES.keys())}"
        )

    def _resolve_solvent_name(self, solvent: str) -> str:
        """
        Resolve a solvent name to its internal identifier.

        Parameters
        ----------
        solvent : str
            Chemical name of the solvent (e.g., 'DCM', 'dichloromethane').

        Returns
        -------
        str
            Internal solvent identifier.
        """
        # Check if it's in the solvent library
        if solvent in self.SOLVENT_LIBRARY:
            identifier = self.SOLVENT_LIBRARY[solvent].get('identifier', solvent)
            full_name = self.SOLVENT_LIBRARY[solvent].get('full_name', solvent)
            print(f"Solvent '{solvent}' ({full_name}) -> {identifier}")
            return identifier

        # Check if it's directly an available solvent (for backwards compatibility)
        if solvent in self.AVAILABLE_SOLVENTS:
            return solvent

        # Not found
        available_names = [k for k, v in self.SOLVENT_LIBRARY.items() if 'full_name' in v]
        raise ValueError(
            f"Solvent '{solvent}' not found.\n"
            f"Available solvent names: {available_names}\n"
            f"Or use internal identifiers: {list(self.AVAILABLE_SOLVENTS.keys())}"
        )

    def _validate_inputs(self):
        """Validate that all input files and parameters exist."""
        self._debug_print("Starting input validation...", level=1)

        # Check input PDB exists
        self._debug_print(f"Checking input PDB file: {self.input_pdb}", level=1)
        if not Path(self.input_pdb).exists():
            self._debug_print(f"ERROR: Input PDB file not found!", level=1)
            raise FileNotFoundError(f"Input PDB file not found: {self.input_pdb}")
        self._debug_print(f"  Input PDB exists: YES", level=1)

        # Check counterions exist in data folder (skip for built-in counterions)
        if self.process_counterions and not self._builtin_counterion:
            self._debug_print(f"Validating counterions: {self.counterions}", level=1)
            for ci in self.counterions:
                self._debug_print(f"  Checking counterion: {ci}", level=1)
                if ci not in self.AVAILABLE_COUNTERIONS:
                    self._debug_print(f"ERROR: Counterion residue '{ci}' not found!", level=1)
                    raise ValueError(f"Counterion residue '{ci}' not found. Available: {list(self.AVAILABLE_COUNTERIONS.keys())}")
                mol2_path = self.counterions_dir / self.AVAILABLE_COUNTERIONS[ci]['mol2']
                self._debug_print(f"    Looking for mol2 file: {mol2_path}", level=1)
                if not mol2_path.exists():
                    self._debug_print(f"ERROR: Counterion mol2 file not found!", level=1)
                    raise FileNotFoundError(f"Counterion mol2 file not found: {mol2_path}")
                self._debug_print(f"    mol2 file exists: YES", level=1)
        elif self._builtin_counterion:
            self._debug_print(f"Using built-in counterion: {self._builtin_counterion_info.get('ion_name', 'Cl-')}", level=1)

        # Check substrate exists
        if self.process_substrate and self.substrate:
            self._debug_print(f"Validating substrate: {self.substrate}", level=1)
            if self.substrate not in self.AVAILABLE_SUBSTRATES:
                self._debug_print(f"ERROR: Substrate residue '{self.substrate}' not found!", level=1)
                raise ValueError(f"Substrate residue '{self.substrate}' not found. Available: {list(self.AVAILABLE_SUBSTRATES.keys())}")
            mol2_path = self.substrates_dir / self.AVAILABLE_SUBSTRATES[self.substrate]['mol2']
            self._debug_print(f"  Looking for mol2 file: {mol2_path}", level=1)
            if not mol2_path.exists():
                self._debug_print(f"ERROR: Substrate mol2 file not found!", level=1)
                raise FileNotFoundError(f"Substrate mol2 file not found: {mol2_path}")
            self._debug_print(f"  mol2 file exists: YES", level=1)

        # Check solvent exists
        if self.solvent:
            self._debug_print(f"Validating solvent: {self.solvent}", level=1)
            if self.solvent not in self.AVAILABLE_SOLVENTS:
                self._debug_print(f"ERROR: Solvent '{self.solvent}' not found!", level=1)
                raise ValueError(f"Solvent '{self.solvent}' not found. Available: {list(self.AVAILABLE_SOLVENTS.keys())}")
            self._debug_print(f"  Solvent '{self.solvent}' is available: YES", level=1)

        self._debug_print("Input validation complete - all checks passed", level=1)

    def _copy_data_files(self):
        """Copy required data files to working directory."""
        print("\nCopying data files to working directory...")
        self._debug_print("Starting _copy_data_files()")
        self._debug_print(f"  Source counterions dir: {self.counterions_dir}")
        self._debug_print(f"  Source substrates dir: {self.substrates_dir}")
        self._debug_print(f"  Source solvent dir: {self.solvent_dir}")
        self._debug_print(f"  Target working dir: {self.working_dir}")

        # Copy counterion files (skip for built-in counterions)
        if self.process_counterions and not self._builtin_counterion:
            self._debug_print(f"Copying counterion files for: {self.counterions}")
            for ci in self.counterions:
                for file_type in ['mol2', 'frcmod']:
                    src = self.counterions_dir / self.AVAILABLE_COUNTERIONS[ci][file_type]
                    dst = self.working_dir / self.AVAILABLE_COUNTERIONS[ci][file_type]
                    self._debug_print(f"  Copying {src} -> {dst}", level=1)
                    if src.exists():
                        shutil.copy2(src, dst)
                        print(f"  Copied {src.name}")
                        self._debug_print(f"    SUCCESS", level=1)
                    else:
                        self._debug_print(f"    WARNING: Source file does not exist!", level=1)

        # Copy substrate files
        if self.process_substrate and self.substrate:
            self._debug_print(f"Copying substrate files for: {self.substrate}")
            for file_type in ['mol2', 'frcmod', 'pdb']:
                if file_type in self.AVAILABLE_SUBSTRATES[self.substrate]:
                    src = self.substrates_dir / self.AVAILABLE_SUBSTRATES[self.substrate][file_type]
                    dst = self.working_dir / self.AVAILABLE_SUBSTRATES[self.substrate][file_type]
                    self._debug_print(f"  Copying {src} -> {dst}", level=1)
                    if src.exists():
                        shutil.copy2(src, dst)
                        print(f"  Copied {src.name}")
                        self._debug_print(f"    SUCCESS", level=1)
                    else:
                        self._debug_print(f"    WARNING: Source file does not exist!", level=1)

        # Copy solvent files (skip for built-in water models)
        if self.solvent:
            solvent_info = self.AVAILABLE_SOLVENTS[self.solvent]
            if solvent_info.get('builtin', False):
                self._debug_print(f"Using built-in water model: {self.solvent} (no lib file needed)")
                print(f"  Using built-in water model: {self.solvent}")
            else:
                self._debug_print(f"Copying solvent files for: {self.solvent}")
                src = self.solvent_dir / solvent_info['lib']
                dst = self.working_dir / solvent_info['lib']
                self._debug_print(f"  Copying {src} -> {dst}", level=1)
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"  Copied {src.name}")
                    self._debug_print(f"    SUCCESS", level=1)
                else:
                    self._debug_print(f"    WARNING: Source file does not exist!", level=1)

                # Copy leaprc.gaff2
                leaprc_src = self.solvent_dir / "leaprc.gaff2"
                leaprc_dst = self.working_dir / "leaprc.gaff2"
                self._debug_print(f"  Copying {leaprc_src} -> {leaprc_dst}", level=1)
                if leaprc_src.exists():
                    shutil.copy2(leaprc_src, leaprc_dst)
                    print(f"  Copied leaprc.gaff2")
                    self._debug_print(f"    SUCCESS", level=1)
                else:
                    self._debug_print(f"    WARNING: leaprc.gaff2 does not exist!", level=1)

        self._debug_print("_copy_data_files() complete")

    # =========================================================================
    # STEP 1: Prepare Topology (from makeAmberVals.py)
    # =========================================================================

    def prepare_topology(self):
        """
        Prepare AMBER topology from input PDB using metallicious.

        This method performs the following steps:
        1. Load the cage structure with metallicious
        2. Prepare initial topology
        3. Parametrize and output to GROMACS format
        4. Convert to AMBER format (prmtop/inpcrd)
        """
        self._debug_print("="*60)
        self._debug_print("Starting prepare_topology()")
        self._debug_print("="*60)

        self._debug_print("Checking required imports...")
        if supramolecular_structure is None:
            self._debug_print("ERROR: metallicious not imported!")
            raise ImportError("metallicious is required for topology preparation. Install with: pip install metallicious")
        self._debug_print("  metallicious: OK")

        if pmd is None:
            self._debug_print("ERROR: parmed not imported!")
            raise ImportError("parmed is required. Install with: pip install parmed")
        self._debug_print("  parmed: OK")

        print("="*70)
        print("STEP 1: PREPARING TOPOLOGY (metallicious)")
        print("="*70)

        # Change to working directory
        original_dir = os.getcwd()
        self._debug_print(f"Original directory: {original_dir}")
        self._debug_print(f"Changing to working directory: {self.working_dir}")
        os.chdir(self.working_dir)

        try:
            # Determine absolute path to input PDB
            if Path(self.input_pdb).is_absolute():
                pdb_path = self.input_pdb
            else:
                pdb_path = str(Path(original_dir) / self.input_pdb)
            self._debug_print(f"Resolved input PDB path: {pdb_path}")
            self._debug_print(f"  Path exists: {Path(pdb_path).exists()}")

            print(f"\nLoading cage structure from {self.input_pdb}...")
            print(f"Metal charges: {self.metal_charges}")
            print(f"LJ type: {self.lj_type}")

            self._debug_print("Creating supramolecular_structure object...")
            self._debug_print(f"  pdb_path: {pdb_path}")
            self._debug_print(f"  metal_charges: {self.metal_charges}")
            self._debug_print(f"  LJ_type: {self.lj_type}")
            try:
                cage = supramolecular_structure(
                    pdb_path,
                    metal_charges=self.metal_charges,
                    LJ_type=self.lj_type
                )
                self._debug_print("  supramolecular_structure created successfully")
            except Exception as e:
                self._debug_print(f"ERROR creating supramolecular_structure: {type(e).__name__}: {e}")
                raise

            self._debug_print("Calling cage.prepare_initial_topology()...")
            try:
                cage.prepare_initial_topology()
                self._debug_print("  prepare_initial_topology() completed")
            except Exception as e:
                self._debug_print(f"ERROR in prepare_initial_topology: {type(e).__name__}: {e}")
                raise

            self._debug_print("Calling cage.parametrize()...")
            self._debug_print("  out_coord: cage_out.pdb")
            self._debug_print("  out_topol: cage_out.top")
            try:
                cage.parametrize(out_coord='cage_out.pdb', out_topol='cage_out.top')
                self._debug_print("  parametrize() completed")
            except Exception as e:
                self._debug_print(f"ERROR in parametrize: {type(e).__name__}: {e}")
                raise

            # Store paths
            self.cage_pdb = self.working_dir / 'cage_out.pdb'
            self.cage_top = self.working_dir / 'cage_out.top'
            self._debug_print(f"Output cage_pdb: {self.cage_pdb}")
            self._debug_print(f"  File exists: {self.cage_pdb.exists()}")
            self._debug_print(f"Output cage_top: {self.cage_top}")
            self._debug_print(f"  File exists: {self.cage_top.exists()}")

            # Load GROMACS topology and structure
            print("\nLoading GROMACS topology...")
            self._debug_print("Loading GROMACS topology with parmed...")
            self._debug_print("  pmd.load_file('cage_out.top', xyz='cage_out.pdb')")
            try:
                gro_top = pmd.load_file('cage_out.top', xyz='cage_out.pdb')
                self._debug_print(f"  Loaded structure with {len(gro_top.atoms)} atoms")
            except Exception as e:
                self._debug_print(f"ERROR loading GROMACS topology: {type(e).__name__}: {e}")
                raise

            # Save as Amber format
            print("Saving Amber topology and coordinates...")
            self._debug_print("Saving to AMBER format...")
            try:
                self._debug_print("  Saving cage_out.prmtop...")
                gro_top.save('cage_out.prmtop', format='amber', overwrite=True)
                self._debug_print("    prmtop saved successfully")

                self._debug_print("  Saving cage_out.inpcrd...")
                gro_top.save('cage_out.inpcrd', format='rst7', overwrite=True)
                self._debug_print("    inpcrd saved successfully")
            except Exception as e:
                self._debug_print(f"ERROR saving AMBER files: {type(e).__name__}: {e}")
                raise

            self.cage_prmtop = self.working_dir / 'cage_out.prmtop'
            self.cage_inpcrd = self.working_dir / 'cage_out.inpcrd'
            self._debug_print(f"Final cage_prmtop: {self.cage_prmtop}")
            self._debug_print(f"  File exists: {self.cage_prmtop.exists()}")
            self._debug_print(f"Final cage_inpcrd: {self.cage_inpcrd}")
            self._debug_print(f"  File exists: {self.cage_inpcrd.exists()}")

            print("\nTopology preparation complete!")
            print(f"  Created: {self.cage_prmtop.name}")
            print(f"  Created: {self.cage_inpcrd.name}")
            self._debug_print("prepare_topology() finished successfully")

        finally:
            self._debug_print(f"Returning to original directory: {original_dir}")
            os.chdir(original_dir)

    # =========================================================================
    # STEP 2: Extract frcmod (from extract_frcmod.py)
    # =========================================================================

    def extract_frcmod(
        self,
        prmtop: Optional[str] = None,
        inpcrd: Optional[str] = None,
        out_mol2: str = "CAGE.mol2",
        out_frcmod: str = "CAGE.frcmod",
        selector: str = ":*",
        use_unique_types: bool = True
    ):
        """
        Extract force field parameters from prmtop to frcmod format.

        Parameters
        ----------
        prmtop : str, optional
            Path to prmtop file. Uses cage_out.prmtop if not specified.
        inpcrd : str, optional
            Path to inpcrd file. Uses cage_out.inpcrd if not specified.
        out_mol2 : str, optional
            Output mol2 file name. Default is "CAGE.mol2".
        out_frcmod : str, optional
            Output frcmod file name. Default is "CAGE.frcmod".
        selector : str, optional
            Atom selection mask. Default is ":*" (all atoms).
        use_unique_types : bool, optional
            If True (default), creates unique atom types for atoms with different
            parameter environments to preserve all metallicious parameters through
            tleap. If False, uses the original behavior where multiple parameters
            for the same atom type combination are collapsed to one.
        """
        self._debug_print("="*60)
        self._debug_print("Starting extract_frcmod()")
        self._debug_print("="*60)
        self._debug_print(f"  prmtop arg: {prmtop}")
        self._debug_print(f"  inpcrd arg: {inpcrd}")
        self._debug_print(f"  out_mol2: {out_mol2}")
        self._debug_print(f"  out_frcmod: {out_frcmod}")
        self._debug_print(f"  selector: {selector}")
        self._debug_print(f"  use_unique_types: {use_unique_types}")

        self._debug_print("Checking parmed import...")
        if pmd is None or AmberMask is None:
            self._debug_print("ERROR: parmed not imported!")
            raise ImportError("parmed is required. Install with: pip install parmed")
        self._debug_print("  parmed: OK")

        print("\n" + "="*70)
        print("STEP 2: EXTRACTING FORCE FIELD PARAMETERS")
        print("="*70)

        # Use default paths if not specified
        prmtop = prmtop or str(self.working_dir / "cage_out.prmtop")
        inpcrd = inpcrd or str(self.working_dir / "cage_out.inpcrd")
        self._debug_print(f"Resolved prmtop path: {prmtop}")
        self._debug_print(f"  File exists: {Path(prmtop).exists()}")
        self._debug_print(f"Resolved inpcrd path: {inpcrd}")
        self._debug_print(f"  File exists: {Path(inpcrd).exists()}")

        print(f"\nInput prmtop: {prmtop}")
        print(f"Input inpcrd: {inpcrd}")
        print(f"Output mol2: {out_mol2}")
        print(f"Output frcmod: {out_frcmod}")
        print(f"Selector: {selector}")
        print(f"Use unique types: {use_unique_types}")

        # Use unique types method if requested (default)
        if use_unique_types:
            if extract_frcmod_unique_types is None:
                print("WARNING: extract_frcmod_unique_types not available, falling back to standard method")
            else:
                print("\nUsing unique atom types to preserve all parameters...")
                original_dir = os.getcwd()
                os.chdir(self.working_dir)
                try:
                    extract_frcmod_unique_types(
                        prmtop=Path(prmtop).name,
                        inpcrd=Path(inpcrd).name,
                        out_mol2=out_mol2,
                        out_frcmod=out_frcmod,
                        selector=selector,
                        verbose=True
                    )
                    self.cage_mol2 = self.working_dir / out_mol2
                    self.cage_frcmod = self.working_dir / out_frcmod
                    print(f"\nWrote: {out_mol2}")
                    print(f"Wrote: {out_frcmod}")
                    self._debug_print("extract_frcmod() with unique types finished successfully")
                    return
                finally:
                    os.chdir(original_dir)

        # Original method (when use_unique_types=False or fallback)
        # Change to working directory
        original_dir = os.getcwd()
        self._debug_print(f"Original directory: {original_dir}")
        self._debug_print(f"Changing to working directory: {self.working_dir}")
        os.chdir(self.working_dir)

        # After changing to working dir, use just the filenames
        prmtop = Path(prmtop).name
        inpcrd = Path(inpcrd).name

        try:
            self._debug_print("Loading structure with parmed...")
            self._debug_print(f"  pmd.load_file({prmtop}, {inpcrd})")
            try:
                struct = pmd.load_file(prmtop, inpcrd)
                self._debug_print(f"  Structure loaded: {len(struct.atoms)} atoms")
            except Exception as e:
                self._debug_print(f"ERROR loading structure: {type(e).__name__}: {e}")
                raise

            self._debug_print("Loading atom info...")
            try:
                struct.load_atom_info()
                self._debug_print("  Atom info loaded successfully")
            except Exception as e:
                self._debug_print(f"ERROR loading atom info: {type(e).__name__}: {e}")
                raise

            # Identify special types
            self._debug_print(f"Collecting special types with selector '{selector}'...")
            try:
                special_types, sel_idx = self._collect_special_types(struct, selector)
                self._debug_print(f"  Found {len(special_types)} special types")
                self._debug_print(f"  Found {len(sel_idx)} atom indices")
            except Exception as e:
                self._debug_print(f"ERROR collecting special types: {type(e).__name__}: {e}")
                raise
            print(f"\nSelector {selector!r}: selected {len(sel_idx)} atoms, {len(special_types)} types")

            # Save subset as mol2
            self._debug_print("Saving subset as mol2 and pdb...")
            sel_idx = sorted(sel_idx)  # Sort indices for consistent ordering
            self._debug_print(f"  Selected {len(sel_idx)} atom indices: {sel_idx[:5]}...{sel_idx[-5:]}")

            # Workaround for parmed bug: when selecting all atoms via list indexing,
            # atoms can be dropped. Use slice syntax when selecting all atoms.
            if len(sel_idx) == len(struct.atoms) and sel_idx == list(range(len(struct.atoms))):
                self._debug_print("  Using slice syntax (all atoms selected)")
                sub = struct[:]  # Use slice to avoid parmed indexing bug
            else:
                sub = struct[sel_idx]
            print(f"Subset structure: {len(sub.atoms)} atoms")
            self._debug_print(f"  Saving {out_mol2}...")
            try:
                sub.save(out_mol2, format="mol2", overwrite=True)
                self._debug_print(f"    {out_mol2} saved successfully")
            except Exception as e:
                self._debug_print(f"ERROR saving mol2: {type(e).__name__}: {e}")
                raise

            out_pdb = out_mol2.rsplit(".", 1)[0] + ".pdb"
            self._debug_print(f"  Saving {out_pdb}...")
            try:
                sub.save(out_pdb, format="pdb", overwrite=True)
                self._debug_print(f"    {out_pdb} saved successfully")
            except Exception as e:
                self._debug_print(f"ERROR saving pdb: {type(e).__name__}: {e}")
                raise

            # Reload structure
            self._debug_print("Reloading structure for parameter extraction...")
            struct = pmd.load_file(prmtop, inpcrd)
            self._debug_print("  Structure reloaded")

            if not special_types:
                self._debug_print("ERROR: No special types identified!")
                raise RuntimeError("No special types identified.")

            def involves_special(types):
                return any(t in special_types for t in types)

            # Collect parameters
            self._debug_print("Collecting force field parameters...")
            type_to_mass = {}
            for a in struct.atoms:
                if a.type in special_types and a.type not in type_to_mass:
                    type_to_mass[a.type] = a.mass
            self._debug_print(f"  Collected {len(type_to_mass)} mass entries")

            bonds = {}
            angles = {}
            diheds = {}
            improps = {}

            # Bonds
            self._debug_print("  Collecting bond parameters...")
            for b in struct.bonds:
                t1, t2 = b.atom1.type, b.atom2.type
                if t1 is None or t2 is None or b.type is None:
                    continue
                if involves_special([t1, t2]):
                    key = tuple(sorted((t1, t2)))
                    bonds[key] = (b.type.k, b.type.req)
            self._debug_print(f"    Found {len(bonds)} bond types")

            # Angles
            self._debug_print("  Collecting angle parameters...")
            for a in struct.angles:
                t1, t2, t3 = a.atom1.type, a.atom2.type, a.atom3.type
                if None in (t1, t2, t3) or a.type is None:
                    continue
                if involves_special([t1, t2, t3]):
                    key = (t1, t2, t3)
                    angles[key] = (a.type.k, a.type.theteq)
            self._debug_print(f"    Found {len(angles)} angle types")

            # Dihedrals
            self._debug_print("  Collecting dihedral parameters...")
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
            self._debug_print(f"    Found {len(diheds)} dihedral types")
            self._debug_print(f"    Found {len(improps)} improper types")

            # Nonbonded parameters
            # AMBER frcmod NONBON section expects R* = Rmin/2 (half the minimum energy distance)
            # parmed's rmin attribute is actually Rmin, so we need to divide by 2
            self._debug_print("  Collecting nonbonded parameters...")
            nonbon = {}
            for a in struct.atoms:
                t = a.type
                if t in special_types and a.atom_type is not None:
                    # Try to get rmin first (preferred for AMBER)
                    rmin = getattr(a.atom_type, "rmin", None)
                    eps = getattr(a.atom_type, "epsilon", None)
                    if rmin is None:
                        # Fall back to sigma and convert to rmin
                        sigma = getattr(a.atom_type, "sigma", None)
                        if sigma is not None:
                            rmin = sigma * (2.0 ** (1.0/6.0))
                    if rmin is None or eps is None:
                        continue
                    # AMBER frcmod expects R* = Rmin/2
                    r_star = rmin / 2.0
                    nonbon[t] = (r_star, eps)
            self._debug_print(f"    Found {len(nonbon)} nonbonded types")

            # Write frcmod
            self._debug_print(f"Writing frcmod file: {out_frcmod}")
            try:
                self._write_frcmod(out_frcmod, type_to_mass, bonds, angles, diheds, improps, nonbon)
                self._debug_print(f"  {out_frcmod} written successfully")
            except Exception as e:
                self._debug_print(f"ERROR writing frcmod: {type(e).__name__}: {e}")
                raise

            self.cage_mol2 = self.working_dir / out_mol2
            self.cage_frcmod = self.working_dir / out_frcmod
            self._debug_print(f"Output files set:")
            self._debug_print(f"  cage_mol2: {self.cage_mol2}")
            self._debug_print(f"  cage_frcmod: {self.cage_frcmod}")

            print(f"\nWrote: {out_mol2}")
            print(f"Wrote: {out_frcmod}")
            print(f"Special types: {' '.join(sorted(special_types))}")
            self._debug_print("extract_frcmod() finished successfully")

        finally:
            self._debug_print(f"Returning to original directory: {original_dir}")
            os.chdir(original_dir)

    def _collect_special_types(self, struct, selector: str) -> Tuple[Set[str], List[int]]:
        """Collect atom types for the given selector mask.

        Returns
        -------
        special_types : Set[str]
            Set of unique atom types in the selection
        indices : List[int]
            List of atom indices (preserves all atoms, no deduplication)
        """
        self._debug_print(f"_collect_special_types() called with selector: '{selector}'", level=1)
        selector = selector.strip()

        self._debug_print(f"  Creating AmberMask...", level=1)
        try:
            mask = AmberMask(struct, selector)
            indices = list(mask.Selected())  # Convert generator to list
            self._debug_print(f"  AmberMask created, {len(indices) if indices else 0} indices selected", level=1)
        except Exception as e:
            self._debug_print(f"  ERROR creating AmberMask: {type(e).__name__}: {e}", level=1)
            raise

        if not indices:
            self._debug_print(f"  ERROR: Mask selected 0 atoms!", level=1)
            raise RuntimeError(f"Mask '{selector}' selected 0 atoms. Check residue names.")

        # Collect atom types directly from indices (avoid set of atoms which can dedupe identical atoms)
        special_types = {struct.atoms[i].type for i in indices if struct.atoms[i].type is not None}
        self._debug_print(f"  Found {len(indices)} atoms", level=1)
        self._debug_print(f"  Found {len(special_types)} special types: {sorted(special_types)[:10]}...", level=1)

        if not special_types:
            self._debug_print(f"  ERROR: No atoms had atom types!", level=1)
            raise RuntimeError(f"Mask '{selector}' selected atoms but none had atom types.")

        return special_types, indices

    @staticmethod
    def _fmt(x, w=12, p=6):
        return f"{x:{w}.{p}f}"

    def _write_frcmod(self, filename, type_to_mass, bonds, angles, diheds, improps, nonbon):
        """Write frcmod file with extracted parameters."""
        with open(filename, "w") as f:
            f.write("EXACT_FROM_PRMTOP\n\n")

            f.write("MASS\n")
            for t in sorted(type_to_mass.keys()):
                f.write(f"{t:<6} {self._fmt(type_to_mass[t], w=10, p=4)}\n")

            f.write("\nBOND\n")
            for (t1, t2), (k, req) in sorted(bonds.items()):
                f.write(f"{t1:<2}-{t2:<2}  {self._fmt(k)}  {self._fmt(req)}\n")

            f.write("\nANGLE\n")
            for (t1, t2, t3), (k, th) in sorted(angles.items()):
                f.write(f"{t1:<2}-{t2:<2}-{t3:<2}  {self._fmt(k)}  {self._fmt(th)}\n")

            f.write("\nDIHE\n")
            for (t1, t2, t3, t4), terms in sorted(diheds.items()):
                for (pk, phase, per) in sorted(terms):
                    f.write(f"{t1:<2}-{t2:<2}-{t3:<2}-{t4:<2}  {self._fmt(pk)}  {self._fmt(phase)}  {self._fmt(per)}\n")

            f.write("\nIMPROPER\n")
            for (t1, t2, t3, t4), terms in sorted(improps.items()):
                for (pk, phase, per) in sorted(terms):
                    f.write(f"{t1:<2}-{t2:<2}-{t3:<2}-{t4:<2}  {self._fmt(pk)}  {self._fmt(phase)}  {self._fmt(per)}\n")

            f.write("\nNONBON\n")
            for t, (r_star, eps) in sorted(nonbon.items()):
                f.write(f"{t:<6} {self._fmt(r_star)}  {self._fmt(eps)}\n")

            f.write("\nEND\n")

    def _get_atom_types_from_frcmod(self, frcmod_path: str) -> dict:
        """
        Extract atom types from frcmod file and infer element mappings.

        Returns a dict mapping atom type -> (element, hybridization).
        Element is inferred from the first character of the type name.
        """
        atom_types = {}

        try:
            with open(frcmod_path, 'r') as f:
                in_mass_section = False
                for line in f:
                    line = line.strip()
                    if line == "MASS":
                        in_mass_section = True
                        continue
                    if in_mass_section:
                        if not line or line.startswith(('BOND', 'ANGLE', 'DIHE', 'IMPROPER', 'NONBON', 'END')):
                            break
                        # Parse atom type from MASS section: "TYPE  MASS"
                        parts = line.split()
                        if len(parts) >= 2:
                            atom_type = parts[0]
                            try:
                                mass = float(parts[1])
                            except ValueError:
                                continue

                            # Infer element from type name and mass
                            first_char = atom_type[0].upper()

                            # Handle special cases
                            if first_char == 'P' and atom_type.upper().startswith('PD'):
                                element = "Pd"
                                hybridization = "sp3"
                            elif first_char == 'C':
                                element = "C"
                                hybridization = "sp2"
                            elif first_char == 'H':
                                element = "H"
                                hybridization = "sp3"
                            elif first_char == 'N':
                                element = "N"
                                hybridization = "sp2"
                            elif first_char == 'O':
                                element = "O"
                                hybridization = "sp2"
                            elif first_char == 'S':
                                element = "S"
                                hybridization = "sp3"
                            elif first_char == 'F':
                                element = "F"
                                hybridization = "sp3"
                            elif first_char == 'B':
                                element = "B"
                                hybridization = "sp3"
                            else:
                                # For unknown elements, use mass to guess
                                if mass < 2:
                                    element = "H"
                                    hybridization = "sp3"
                                elif mass < 14:
                                    element = "C"
                                    hybridization = "sp2"
                                elif mass < 18:
                                    element = "N"
                                    hybridization = "sp2"
                                elif mass < 20:
                                    element = "O"
                                    hybridization = "sp2"
                                else:
                                    element = first_char
                                    hybridization = "sp3"

                            atom_types[atom_type] = (element, hybridization)
        except Exception as e:
            self._debug_print(f"Warning: Could not read frcmod for atom types: {e}")

        return atom_types

    def _generate_add_atom_types_block(self, frcmod_path: str) -> str:
        """
        Generate tleap addAtomTypes block for custom atom types.

        This is required to register custom atom types (like c0, n1, etc.)
        with tleap before loading frcmod/mol2 files.
        """
        atom_types = self._get_atom_types_from_frcmod(frcmod_path)

        if not atom_types:
            return ""

        lines = ["# Register custom atom types BEFORE loading frcmod"]
        lines.append("# This is critical - without this, tleap doesn't know what element each type is")
        lines.append("addAtomTypes {")

        for atom_type, (element, hybridization) in sorted(atom_types.items()):
            lines.append(f'    {{ "{atom_type}"  "{element}" "{hybridization}" }}')

        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # STEP 3: Position Molecules (from position_molecules.py)
    # =========================================================================

    def position_molecules(self, input_pdb: Optional[str] = None, output_pdb: Optional[str] = None):
        """
        Position counterions and substrate around the cage structure.

        Parameters
        ----------
        input_pdb : str, optional
            Input PDB file with cage structure. Uses CAGE.pdb if not specified.
        output_pdb : str, optional
            Output PDB file name. Default is based on input name with _positioned suffix.
        """
        self._debug_print("="*60)
        self._debug_print("Starting position_molecules()")
        self._debug_print("="*60)
        self._debug_print(f"  input_pdb arg: {input_pdb}")
        self._debug_print(f"  output_pdb arg: {output_pdb}")

        print("\n" + "="*70)
        print("STEP 3: POSITIONING MOLECULES")
        print("="*70)

        # Use default paths if not specified
        if input_pdb is None:
            input_pdb = str(self.working_dir / "CAGE.pdb")
            self._debug_print(f"  Using default input_pdb: {input_pdb}")

        if output_pdb is None:
            base_name = Path(input_pdb).stem
            output_pdb = f"{base_name}_positioned.pdb"
            self._debug_print(f"  Using default output_pdb: {output_pdb}")

        self._debug_print(f"Checking input file exists: {input_pdb}")
        self._debug_print(f"  File exists: {Path(input_pdb).exists()}")

        # Change to working directory
        original_dir = os.getcwd()
        self._debug_print(f"Original directory: {original_dir}")
        self._debug_print(f"Changing to working directory: {self.working_dir}")
        os.chdir(self.working_dir)

        # After changing to working dir, use just the filename
        input_pdb = Path(input_pdb).name

        try:
            print(f"\nReading {input_pdb}...")
            self._debug_print(f"Reading PDB file: {input_pdb}")
            try:
                pdb_residues = self._read_pdb(input_pdb)
                self._debug_print(f"  Read {len(pdb_residues)} residues")
                for resid, atoms in sorted(pdb_residues.items()):
                    self._debug_print(f"    Residue {resid}: {atoms[0].resname if atoms else 'EMPTY'} ({len(atoms)} atoms)", level=1)
            except Exception as e:
                self._debug_print(f"ERROR reading PDB: {type(e).__name__}: {e}")
                raise
            print(f"  Found {len(pdb_residues)} residues")

            # Calculate cage bounds
            print("\nAnalyzing cage structure...")
            self._debug_print("Calculating cage bounds...")
            try:
                cage_center, min_bounds, max_bounds, max_radius, pd1_coords, pd2_coords, linker_coords = self._calculate_cage_bounds(pdb_residues)
                self._debug_print(f"  Cage center: {cage_center}")
                self._debug_print(f"  Min bounds: {min_bounds}")
                self._debug_print(f"  Max bounds: {max_bounds}")
                self._debug_print(f"  Max radius: {max_radius}")
                self._debug_print(f"  Pd1 coords: {pd1_coords}")
                self._debug_print(f"  Pd2 coords: {pd2_coords}")
                self._debug_print(f"  Linker coords shape: {linker_coords.shape}")
            except Exception as e:
                self._debug_print(f"ERROR calculating cage bounds: {type(e).__name__}: {e}")
                raise
            print(f"  Cage center: ({cage_center[0]:.3f}, {cage_center[1]:.3f}, {cage_center[2]:.3f})")
            print(f"  Cage radius: {max_radius:.3f} A")

            # Process counterions
            counterion_data = None
            counterion_resids = None
            if self.process_counterions and self.counterions:
                self._debug_print(f"Processing counterions: {self.counterions}")
                self._debug_print(f"  Counterion resids: {self._counterion_resids}")
                self._debug_print(f"  Placement multiplier: {self.counterion_placement_multiplier}")
                try:
                    counterion_data, counterion_resids = self._process_counterions(
                        cage_center, max_radius,
                        counterion_names=self.counterions,
                        new_resids=self._counterion_resids,
                        placement_multiplier=self.counterion_placement_multiplier
                    )
                    self._debug_print(f"  Counterion processing complete")
                    self._debug_print(f"  counterion_data keys: {list(counterion_data.keys()) if counterion_data else None}")
                except Exception as e:
                    self._debug_print(f"ERROR processing counterions: {type(e).__name__}: {e}")
                    raise
            elif self._builtin_counterion:
                self._debug_print(f"Using built-in counterion: {self._builtin_counterion_info.get('ion_name', 'Cl-')}")
                print(f"  Using built-in counterion: {self._builtin_counterion_info.get('ion_name', 'Cl-')} (handled by tleap addIons)")
            else:
                self._debug_print("Skipping counterion processing (disabled or no counterions)")

            # Process substrate
            substrate_atoms = None
            if self.process_substrate and self.substrate:
                self._debug_print(f"Processing substrate: {self.substrate}")
                self._debug_print(f"  Substrate resid: {self._substrate_resid}")
                self._debug_print(f"  Covalent cutoff: {self.substrate_covalent_cutoff}")
                try:
                    substrate_atoms, _, _, _ = self._process_substrate(
                        cage_center, pd1_coords, pd2_coords, linker_coords,
                        substrate_name=self.substrate,
                        substrate_resid=self._substrate_resid,
                        covalent_cutoff=self.substrate_covalent_cutoff
                    )
                    self._debug_print(f"  Substrate processing complete")
                    self._debug_print(f"  substrate_atoms count: {len(substrate_atoms) if substrate_atoms else 0}")
                except Exception as e:
                    self._debug_print(f"ERROR processing substrate: {type(e).__name__}: {e}")
                    raise
            else:
                self._debug_print("Skipping substrate processing (disabled or no substrate)")

            # Build complete structure
            self._debug_print("Building complete structure...")
            try:
                complete_pdb_residues = self._build_complete_structure(
                    pdb_residues,
                    counterion_data=counterion_data,
                    counterion_resids=counterion_resids,
                    substrate_atoms=substrate_atoms,
                    substrate_name=self.substrate,
                    substrate_resid=self._substrate_resid
                )
                self._debug_print(f"  Complete structure has {len(complete_pdb_residues)} residues")
            except Exception as e:
                self._debug_print(f"ERROR building complete structure: {type(e).__name__}: {e}")
                raise

            # Write complete PDB file
            print(f"\nWriting complete PDB file to {output_pdb}...")
            self._debug_print(f"Writing PDB file: {output_pdb}")
            try:
                self._write_pdb(output_pdb, complete_pdb_residues)
                self._debug_print(f"  PDB file written successfully")
            except Exception as e:
                self._debug_print(f"ERROR writing PDB: {type(e).__name__}: {e}")
                raise
            self.final_pdb = self.working_dir / output_pdb
            self._debug_print(f"  final_pdb set to: {self.final_pdb}")
            print(f"  Updated {output_pdb}")

            # Print summary
            self._print_summary(complete_pdb_residues)
            self._debug_print("position_molecules() finished successfully")

        finally:
            self._debug_print(f"Returning to original directory: {original_dir}")
            os.chdir(original_dir)

    def _read_pdb(self, filename: str) -> Dict[int, List[Atom]]:
        """Read PDB file and return atoms grouped by residue."""
        self._debug_print(f"_read_pdb() reading: {filename}", level=1)
        residues = {}
        line_count = 0
        atom_count = 0
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('HETATM') or line.startswith('ATOM'):
                        try:
                            serial = int(line[6:11])
                            name = line[12:16].strip()
                            resname = line[17:20].strip()
                            print(f'Current resname: {resname}')
                            resid = int(line[22:26])
                            print(f'Resid: {resid}')
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            occupancy = float(line[54:60]) if len(line) > 54 else 1.0
                            tempfactor = float(line[60:66]) if len(line) > 60 else 0.0

                            atom = Atom(serial, name, resname, resid, x, y, z, occupancy, tempfactor)

                            if resid not in residues:
                                print(f'Current residue: {resid}')
                                residues[resid] = []
                            residues[resid].append(atom)
                            atom_count += 1
                        except Exception as e:
                            self._debug_print(f"  WARNING: Error parsing line {line_count}: {e}", level=1)
                            self._debug_print(f"    Line content: {line.strip()[:60]}...", level=1)
                    line_count += 1

            self._debug_print(f"  Read {line_count} lines, {atom_count} atoms, {len(residues)} residues", level=1)
        except FileNotFoundError:
            self._debug_print(f"  ERROR: File not found: {filename}", level=1)
            raise
        except Exception as e:
            self._debug_print(f"  ERROR reading file: {type(e).__name__}: {e}", level=1)
            raise

        return residues

    def _read_mol2(self, filename: str) -> Tuple[List[Mol2Atom], List[str], Dict, List[str]]:
        """Read mol2 file and return atoms and bonds."""
        self._debug_print(f"_read_mol2() reading: {filename}", level=1)
        atoms = []
        bonds = []
        molecule_info = {}

        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            self._debug_print(f"  Read {len(lines)} lines", level=1)
        except FileNotFoundError:
            self._debug_print(f"  ERROR: File not found: {filename}", level=1)
            raise
        except Exception as e:
            self._debug_print(f"  ERROR reading file: {type(e).__name__}: {e}", level=1)
            raise

        in_molecule = False
        in_atom = False
        in_bond = False

        for i, line in enumerate(lines):
            if '@<TRIPOS>MOLECULE' in line:
                in_molecule = True
                in_atom = False
                in_bond = False
                molecule_info['name'] = lines[i+1].strip()
                self._debug_print(f"  Found molecule: {molecule_info['name']}", level=1)
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
                    try:
                        atom = Mol2Atom(
                            int(parts[0]), parts[1],
                            float(parts[2]), float(parts[3]), float(parts[4]),
                            parts[5], int(parts[6]), parts[7], float(parts[8])
                        )
                        atoms.append(atom)
                    except Exception as e:
                        self._debug_print(f"  WARNING: Error parsing atom at line {i+1}: {e}", level=1)

            elif in_bond and line.strip():
                bonds.append(line)

        self._debug_print(f"  Parsed {len(atoms)} atoms, {len(bonds)} bonds", level=1)
        return atoms, bonds, molecule_info, lines

    def _write_mol2(self, filename: str, atoms: List[Mol2Atom], bonds: List[str],
                    molecule_info: Dict, original_lines: List[str]):
        """Write mol2 file with updated atoms."""
        with open(filename, 'w') as f:
            in_atom = False
            in_bond = False

            for line in original_lines:
                if '@<TRIPOS>ATOM' in line:
                    in_atom = True
                    in_bond = False
                    f.write(line)
                    for atom in atoms:
                        f.write(atom.to_mol2_line())
                    continue
                elif '@<TRIPOS>BOND' in line:
                    in_bond = True
                    in_atom = False
                    f.write(line)
                    for bond in bonds:
                        f.write(bond)
                    continue
                elif '@<TRIPOS>SUBSTRUCTURE' in line:
                    in_bond = False
                    in_atom = False

                if not in_atom and not in_bond:
                    f.write(line)

    def _write_pdb(self, filename: str, residues_dict: Dict[int, List[Atom]]):
        """Write PDB file from residues dictionary."""
        with open(filename, 'w') as f:
            f.write("REMARK, BUILD BY AUTOBIND\n")

            atom_serial = 1
            for resid in sorted(residues_dict.keys()):
                atoms = residues_dict[resid]
                for atom in atoms:
                    original_serial = atom.serial
                    atom.serial = atom_serial
                    f.write(atom.to_pdb_line())
                    atom.serial = original_serial
                    atom_serial += 1
                f.write("TER\n")

            f.write("END\n")

    def _calculate_cage_bounds(self, residues: Dict[int, List[Atom]]) -> Tuple:
        """Calculate the bounds of the cage structure."""
        self._debug_print("_calculate_cage_bounds() called", level=1)
        self._debug_print(f"  Available residue IDs: {sorted(residues.keys())}", level=1)

        # Identify metal atoms by residue name (PD*, Pd*) or atom name containing metal symbols
        metal_patterns = ['PD', 'Pd', 'PT', 'Pt', 'AU', 'Au', 'AG', 'Ag', 'CU', 'Cu', 'ZN', 'Zn', 'FE', 'Fe']
        metal_coords = []
        linker_coords = []

        self._debug_print("  Identifying metals and linkers by residue/atom names...", level=1)
        for resid, atoms in sorted(residues.items()):
            if not atoms:
                continue
            resname = atoms[0].resname.strip().upper()
            self._debug_print(f"    Residue {resid} ({resname}): {len(atoms)} atoms", level=1)

            # Check if this is a metal residue
            is_metal = False
            for pattern in metal_patterns:
                if resname.startswith(pattern.upper()):
                    is_metal = True
                    break

            if is_metal:
                # Metal residue - take the first atom (usually the metal itself)
                metal_coords.append(atoms[0].coords())
                self._debug_print(f"      -> Metal residue, coords: {atoms[0].coords()}", level=1)
            else:
                # Linker residue - add all atom coordinates
                for atom in atoms:
                    linker_coords.append(atom.coords())
                self._debug_print(f"      -> Linker residue", level=1)

        # Validate we found exactly 2 metals (for Pd2L4 cage)
        if len(metal_coords) < 2:
            self._debug_print(f"  ERROR: Found only {len(metal_coords)} metal atoms, expected 2!", level=1)
            raise ValueError(f"Expected 2 metal atoms, found {len(metal_coords)}. Check residue naming.")
        if len(metal_coords) > 2:
            self._debug_print(f"  WARNING: Found {len(metal_coords)} metal atoms, using first 2", level=1)

        pd1_coords = np.array(metal_coords[0])
        pd2_coords = np.array(metal_coords[1])
        self._debug_print(f"  Metal 1 coords: {pd1_coords}", level=1)
        self._debug_print(f"  Metal 2 coords: {pd2_coords}", level=1)

        if not linker_coords:
            self._debug_print("  ERROR: No linker coordinates found!", level=1)
            raise ValueError("No linker coordinates found")

        linker_coords = np.array(linker_coords)
        self._debug_print(f"  Total linker atoms: {len(linker_coords)}", level=1)

        # Calculate cage center
        cage_center = (pd1_coords + pd2_coords) / 2
        self._debug_print(f"  Cage center: {cage_center}", level=1)

        # Calculate cage bounds
        all_cage_coords = np.vstack([linker_coords, [pd1_coords, pd2_coords]])
        min_bounds = np.min(all_cage_coords, axis=0)
        max_bounds = np.max(all_cage_coords, axis=0)
        self._debug_print(f"  Min bounds: {min_bounds}", level=1)
        self._debug_print(f"  Max bounds: {max_bounds}", level=1)

        # Calculate max distance from center
        distances = np.linalg.norm(all_cage_coords - cage_center, axis=1)
        max_radius = np.max(distances)
        self._debug_print(f"  Max radius: {max_radius:.3f} A", level=1)

        return cage_center, min_bounds, max_bounds, max_radius, pd1_coords, pd2_coords, linker_coords

    def _position_counterions_outside(self, cage_center: np.ndarray, max_radius: float,
                                       counterion_atoms_list: List[List[Mol2Atom]],
                                       placement_multiplier: float = 2.0) -> List[List[Mol2Atom]]:
        """Position counterions outside the cage in a tetrahedral arrangement."""
        tetrahedron_directions = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ])
        tetrahedron_directions = tetrahedron_directions / np.linalg.norm(tetrahedron_directions, axis=1)[:, np.newaxis]

        placement_distance = max_radius * placement_multiplier
        positioned_counterions = []

        for i, atoms in enumerate(counterion_atoms_list):
            coords = np.array([atom.coords() for atom in atoms])
            com = np.mean(coords, axis=0)
            new_center = cage_center + tetrahedron_directions[i] * placement_distance
            translation = new_center - com

            new_atoms = []
            for atom in atoms:
                new_atom = copy.deepcopy(atom)
                new_atom.set_coords(atom.coords() + translation)
                new_atoms.append(new_atom)

            positioned_counterions.append(new_atoms)

        return positioned_counterions

    def _position_substrate_inside(self, cage_center: np.ndarray, pd1_coords: np.ndarray,
                                    pd2_coords: np.ndarray, linker_coords: np.ndarray,
                                    substrate_atoms: List[Mol2Atom],
                                    covalent_cutoff: float = 2.5) -> List[Mol2Atom]:
        """Position substrate inside the cage."""
        coords = np.array([atom.coords() for atom in substrate_atoms])
        substrate_com = np.mean(coords, axis=0)

        pd_vector = pd2_coords - pd1_coords
        pd_vector_norm = pd_vector / np.linalg.norm(pd_vector)

        best_position = cage_center.copy()
        best_min_distance = 0

        for offset_along in np.linspace(-3, 3, 13):
            for theta in np.linspace(0, 2*np.pi, 12):
                for r in np.linspace(0, 2.5, 6):
                    perp_vector1 = np.array([pd_vector_norm[1], -pd_vector_norm[0], 0])
                    if np.linalg.norm(perp_vector1) < 0.1:
                        perp_vector1 = np.array([0, pd_vector_norm[2], -pd_vector_norm[1]])
                    perp_vector1 = perp_vector1 / np.linalg.norm(perp_vector1)
                    perp_vector2 = np.cross(pd_vector_norm, perp_vector1)

                    circular_offset = r * (np.cos(theta) * perp_vector1 + np.sin(theta) * perp_vector2)
                    target_position = cage_center + offset_along * pd_vector_norm + circular_offset

                    translation = target_position - substrate_com
                    translated_coords = coords + translation

                    min_linker_distance = float('inf')
                    for linker_coord in linker_coords:
                        distances = np.linalg.norm(translated_coords - linker_coord, axis=1)
                        min_dist = np.min(distances)
                        if min_dist < min_linker_distance:
                            min_linker_distance = min_dist

                    if min_linker_distance > best_min_distance:
                        best_min_distance = min_linker_distance
                        best_position = target_position.copy()

                    if min_linker_distance > covalent_cutoff:
                        best_position = target_position
                        break
                if best_min_distance > covalent_cutoff:
                    break
            if best_min_distance > covalent_cutoff:
                break

        translation = best_position - substrate_com
        positioned_substrate = []
        for atom in substrate_atoms:
            new_atom = copy.deepcopy(atom)
            new_atom.set_coords(atom.coords() + translation)
            positioned_substrate.append(new_atom)

        print(f"   Best position found with min distance to linkers: {best_min_distance:.3f} A")
        return positioned_substrate

    def _update_residue_numbers(self, atoms: List, new_resid: int, new_resname: Optional[str] = None):
        """Update residue numbers in a list of atoms."""
        for atom in atoms:
            atom.resid = new_resid
            if new_resname:
                atom.resname = new_resname

    def _mol2_to_pdb_atoms(self, mol2_atoms: List[Mol2Atom], resid: int,
                           resname: str, starting_serial: int) -> List[Atom]:
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

    def _process_counterions(self, cage_center: np.ndarray, max_radius: float,
                             counterion_names: List[str], new_resids: List[int],
                             placement_multiplier: float = 2.0) -> Tuple[Dict, List[int]]:
        """Position counterions outside the cage."""
        self._debug_print("_process_counterions() called", level=1)
        self._debug_print(f"  counterion_names: {counterion_names}", level=1)
        self._debug_print(f"  new_resids: {new_resids}", level=1)
        self._debug_print(f"  placement_multiplier: {placement_multiplier}", level=1)

        print("\n" + "-"*50)
        print("COUNTERION POSITIONING")
        print("-"*50)

        print("\n1. Reading counterion mol2 files...")
        self._debug_print("Reading counterion mol2 files...", level=1)
        counterion_data = {}

        for name in counterion_names:
            mol2_file = f'{name}.mol2'
            self._debug_print(f"  Reading {mol2_file}...", level=1)
            try:
                atoms, bonds, mol_info, orig_lines = self._read_mol2(mol2_file)
                counterion_data[name] = {
                    'atoms': atoms,
                    'bonds': bonds,
                    'mol_info': mol_info,
                    'orig_lines': orig_lines
                }
                self._debug_print(f"    Read {len(atoms)} atoms, {len(bonds)} bonds", level=1)
                print(f"   {name}: {len(atoms)} atoms")
            except Exception as e:
                self._debug_print(f"  ERROR reading {mol2_file}: {type(e).__name__}: {e}", level=1)
                raise

        print(f"\n2. Positioning counterions (distance = {placement_multiplier}x radius)...")
        self._debug_print(f"Positioning counterions at {placement_multiplier}x radius...", level=1)
        counterion_atoms_list = [counterion_data[name]['atoms'] for name in counterion_names]
        try:
            positioned_counterions = self._position_counterions_outside(
                cage_center, max_radius, counterion_atoms_list, placement_multiplier
            )
            self._debug_print(f"  Positioned {len(positioned_counterions)} counterions", level=1)
        except Exception as e:
            self._debug_print(f"  ERROR positioning counterions: {type(e).__name__}: {e}", level=1)
            raise

        print("\n3. Updating counterion residue numbers...")
        self._debug_print("Updating residue numbers...", level=1)
        for i, (name, new_resid) in enumerate(zip(counterion_names, new_resids)):
            self._update_residue_numbers(positioned_counterions[i], new_resid, name)
            counterion_data[name]['atoms'] = positioned_counterions[i]
            self._debug_print(f"  {name} -> Residue {new_resid}", level=1)
            print(f"   {name} -> Residue {new_resid}")

        print("\n4. Writing updated counterion mol2 files...")
        self._debug_print("Writing updated mol2 files...", level=1)
        for name in counterion_names:
            mol2_file = f'{name}.mol2'
            self._debug_print(f"  Writing {mol2_file}...", level=1)
            try:
                self._write_mol2(
                    mol2_file,
                    counterion_data[name]['atoms'],
                    counterion_data[name]['bonds'],
                    counterion_data[name]['mol_info'],
                    counterion_data[name]['orig_lines']
                )
                self._debug_print(f"    Written successfully", level=1)
                print(f"   Updated {name}.mol2")
            except Exception as e:
                self._debug_print(f"  ERROR writing {mol2_file}: {type(e).__name__}: {e}", level=1)
                raise

        self._debug_print("_process_counterions() complete", level=1)
        return counterion_data, new_resids

    def _process_substrate(self, cage_center: np.ndarray, pd1_coords: np.ndarray,
                           pd2_coords: np.ndarray, linker_coords: np.ndarray,
                           substrate_name: str, substrate_resid: int,
                           covalent_cutoff: float = 2.5) -> Tuple:
        """Position substrate inside the cage."""
        self._debug_print("_process_substrate() called", level=1)
        self._debug_print(f"  substrate_name: {substrate_name}", level=1)
        self._debug_print(f"  substrate_resid: {substrate_resid}", level=1)
        self._debug_print(f"  covalent_cutoff: {covalent_cutoff}", level=1)

        print("\n" + "-"*50)
        print("SUBSTRATE POSITIONING")
        print("-"*50)

        print(f"\n1. Reading {substrate_name} mol2 file...")
        mol2_file = f'{substrate_name}.mol2'
        self._debug_print(f"Reading {mol2_file}...", level=1)
        try:
            substrate_atoms, substrate_bonds, substrate_mol_info, substrate_orig_lines = self._read_mol2(mol2_file)
            self._debug_print(f"  Read {len(substrate_atoms)} atoms, {len(substrate_bonds)} bonds", level=1)
            print(f"   {substrate_name}: {len(substrate_atoms)} atoms")
        except Exception as e:
            self._debug_print(f"  ERROR reading {mol2_file}: {type(e).__name__}: {e}", level=1)
            raise

        print(f"\n2. Positioning {substrate_name} inside cage...")
        self._debug_print("Positioning substrate inside cage...", level=1)
        try:
            positioned_substrate = self._position_substrate_inside(
                cage_center, pd1_coords, pd2_coords, linker_coords,
                substrate_atoms, covalent_cutoff=covalent_cutoff
            )
            self._debug_print(f"  Positioned {len(positioned_substrate)} atoms", level=1)
        except Exception as e:
            self._debug_print(f"  ERROR positioning substrate: {type(e).__name__}: {e}", level=1)
            raise

        print(f"\n3. Updating {substrate_name} residue number to {substrate_resid}...")
        self._debug_print(f"Updating residue number to {substrate_resid}...", level=1)
        self._update_residue_numbers(positioned_substrate, substrate_resid, substrate_name)
        self._debug_print("  Residue numbers updated", level=1)

        print(f"\n4. Writing updated {substrate_name} mol2 file...")
        self._debug_print(f"Writing {mol2_file}...", level=1)
        try:
            self._write_mol2(mol2_file, positioned_substrate,
                            substrate_bonds, substrate_mol_info, substrate_orig_lines)
            self._debug_print("  Written successfully", level=1)
            print(f"   Updated {substrate_name}.mol2")
        except Exception as e:
            self._debug_print(f"  ERROR writing {mol2_file}: {type(e).__name__}: {e}", level=1)
            raise

        self._debug_print("_process_substrate() complete", level=1)
        return positioned_substrate, substrate_bonds, substrate_mol_info, substrate_orig_lines

    def _build_complete_structure(self, pdb_residues: Dict[int, List[Atom]],
                                   counterion_data: Optional[Dict] = None,
                                   counterion_resids: Optional[List[int]] = None,
                                   substrate_atoms: Optional[List[Mol2Atom]] = None,
                                   substrate_name: str = 'PZQ',
                                   substrate_resid: Optional[int] = 11) -> Dict[int, List[Atom]]:
        """Build complete PDB structure."""
        print("\n" + "-"*50)
        print("BUILDING COMPLETE STRUCTURE")
        print("-"*50)

        complete_pdb_residues = copy.deepcopy(pdb_residues)
        last_serial = max([atom.serial for residue in pdb_residues.values() for atom in residue])

        if counterion_data is not None and counterion_resids is not None:
            print("\nAdding counterions to structure...")
            counterion_names = list(counterion_data.keys())
            for i, (name, new_resid) in enumerate(zip(counterion_names, counterion_resids)):
                pdb_atoms = self._mol2_to_pdb_atoms(
                    counterion_data[name]['atoms'],
                    new_resid, name, last_serial + 1
                )
                complete_pdb_residues[new_resid] = pdb_atoms
                last_serial += len(pdb_atoms)
                print(f"  Added {name} as residue {new_resid}")

        if substrate_atoms is not None:
            print(f"\nAdding {substrate_name} to structure...")
            substrate_pdb_atoms = self._mol2_to_pdb_atoms(
                substrate_atoms, substrate_resid, substrate_name, last_serial + 1
            )
            complete_pdb_residues[substrate_resid] = substrate_pdb_atoms
            print(f"  Added {substrate_name} as residue {substrate_resid}")

        return complete_pdb_residues

    def _print_summary(self, complete_pdb_residues: Dict[int, List[Atom]]):
        """Print summary of the final structure."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total residues in final structure: {len(complete_pdb_residues)}")

        for resid in sorted(complete_pdb_residues.keys()):
            atoms = complete_pdb_residues[resid]
            resname = atoms[0].resname if atoms else "UNK"
            print(f"  {resid:2d}: {resname} ({len(atoms)} atoms)")

    # =========================================================================
    # STEP 4: Generate and Run tleap
    # =========================================================================

    def generate_tleap_input(self, output_file: str = "input_tleap.in") -> str:
        """
        Generate tleap input file based on configured components.

        Parameters
        ----------
        output_file : str, optional
            Output filename for tleap input. Default is "input_tleap.in".

        Returns
        -------
        str
            Path to the generated tleap input file.
        """
        self._debug_print("="*60)
        self._debug_print("Starting generate_tleap_input()")
        self._debug_print("="*60)
        self._debug_print(f"  output_file: {output_file}")

        print("\n" + "="*70)
        print("STEP 4: GENERATING TLEAP INPUT")
        print("="*70)

        tleap_path = self.working_dir / output_file
        self._debug_print(f"  tleap_path: {tleap_path}")

        # Build component lists
        self._debug_print("Building component lists...")
        components = ['CAGE']
        mol2_loads = ['CAGE = loadmol2 CAGE.mol2']
        frcmod_loads = ['loadamberparams CAGE.frcmod']

        # Verify CAGE files exist
        cage_mol2 = self.working_dir / "CAGE.mol2"
        cage_frcmod = self.working_dir / "CAGE.frcmod"
        self._debug_print(f"  Checking CAGE.mol2: {cage_mol2.exists()}")
        self._debug_print(f"  Checking CAGE.frcmod: {cage_frcmod.exists()}")

        # Add substrate
        if self.process_substrate and self.substrate:
            self._debug_print(f"  Adding substrate: {self.substrate}")
            components.append(self.substrate)
            mol2_loads.append(f'{self.substrate} = loadmol2 {self.substrate}.mol2')
            frcmod_loads.append(f'loadamberparams {self.substrate}.frcmod')
            # Verify substrate files
            sub_mol2 = self.working_dir / f"{self.substrate}.mol2"
            sub_frcmod = self.working_dir / f"{self.substrate}.frcmod"
            self._debug_print(f"    {self.substrate}.mol2 exists: {sub_mol2.exists()}")
            self._debug_print(f"    {self.substrate}.frcmod exists: {sub_frcmod.exists()}")

        # Add counterions (skip for built-in counterions like Cl-)
        if self.process_counterions and not self._builtin_counterion:
            self._debug_print(f"  Adding counterions: {self.counterions}")
            for ci in self.counterions:
                components.append(ci)
                mol2_loads.append(f'{ci} = loadmol2 {ci}.mol2')
                frcmod_loads.append(f'loadamberparams {ci}.frcmod')
                # Verify counterion files
                ci_mol2 = self.working_dir / f"{ci}.mol2"
                ci_frcmod = self.working_dir / f"{ci}.frcmod"
                self._debug_print(f"    {ci}.mol2 exists: {ci_mol2.exists()}")
                self._debug_print(f"    {ci}.frcmod exists: {ci_frcmod.exists()}")
        elif self._builtin_counterion:
            self._debug_print(f"  Using built-in counterion: {self._builtin_counterion_info.get('ion_name', 'Cl-')}")

        # Build combination list starting with CAGE (loaded from CAGE.mol2)
        # The cage is loaded as a single unit, not individual residues
        combine_list = ['CAGE']
        self._debug_print(f"  Base combine list: {combine_list}")

        if self.process_substrate and self.substrate:
            combine_list.append(self.substrate)
        if self.process_counterions and not self._builtin_counterion:
            combine_list.extend(self.counterions)
        self._debug_print(f"  Final combine list: {combine_list}")

        # Solvent configuration
        solvent_info = self.AVAILABLE_SOLVENTS.get(self.solvent, {})
        is_builtin_water = solvent_info.get('builtin', False)
        solvent_box = solvent_info.get('box_name', 'DCMequilbox')
        solvent_lib = solvent_info.get('lib', 'DCMequilbox.lib') if not is_builtin_water else None
        water_leaprc = solvent_info.get('leaprc', 'leaprc.water.opc')
        self._debug_print(f"  Solvent: {self.solvent}")
        self._debug_print(f"    builtin water: {is_builtin_water}")
        self._debug_print(f"    box_name: {solvent_box}")
        if not is_builtin_water:
            self._debug_print(f"    lib: {solvent_lib}")
            # Verify solvent lib exists
            solvent_lib_path = self.working_dir / solvent_lib
            self._debug_print(f"    {solvent_lib} exists: {solvent_lib_path.exists()}")

        # Write tleap input
        self._debug_print(f"Writing tleap input file: {tleap_path}")
        try:
            with open(tleap_path, 'w') as f:
                f.write("# AutoBind generated tleap input\n")
                f.write("source leaprc.gaff2\n")
                f.write(f"source {water_leaprc}\n\n")

                # Add custom atom types BEFORE loading frcmod
                # This is critical - without this, tleap doesn't know what element each type is
                add_atom_types_block = self._generate_add_atom_types_block(str(cage_frcmod))
                if add_atom_types_block:
                    f.write(add_atom_types_block)
                    f.write("\n")
                    self._debug_print("  Added addAtomTypes block for custom cage atom types")

                # Load frcmod parameters FIRST to define atom types
                f.write("# Load frcmod parameters (must be before mol2 to define atom types)\n")
                for load in frcmod_loads:
                    f.write(f"{load}\n")
                    self._debug_print(f"    Writing: {load}", level=1)

                f.write("\n# Load mol2 files\n")
                for load in mol2_loads:
                    f.write(f"{load}\n")
                    self._debug_print(f"    Writing: {load}", level=1)

                f.write("\n# Combine all components\n")
                combine_cmd = f"mol = combine {{ {' '.join(combine_list)} }}"
                f.write(f"{combine_cmd}\n")
                self._debug_print(f"    Writing: {combine_cmd}", level=1)
                f.write("check mol\n\n")

                f.write("# Save dry system\n")
                f.write("savepdb mol ALL_dry.pdb\n")
                f.write("saveamberparm mol ALL_dry.prmtop ALL_dry.inpcrd\n\n")

                f.write("# Solvate\n")
                if not is_builtin_water:
                    f.write(f"loadoff {solvent_lib}\n")
                f.write(f"solvateBox mol {solvent_box} 12.0\n")

                # Add counterions
                if self._builtin_counterion:
                    # Use built-in counterion (e.g., Cl-) with specified count
                    ion_name = self._builtin_counterion_info.get('ion_name', 'Cl-')
                    ion_count = self._builtin_counterion_info.get('count', 4)
                    f.write(f"# Add {ion_count} {ion_name} counterions\n")
                    f.write(f"addions mol {ion_name} {ion_count}\n")
                    self._debug_print(f"    Writing: addions mol {ion_name} {ion_count}", level=1)

                # Neutralize system
                f.write("addions mol Na+ 0\n")
                f.write("addions mol Cl- 0\n\n")

                f.write("# Save solvated system\n")
                f.write("savepdb mol CAGE_solv.pdb\n")
                f.write("saveamberparm mol CAGE_solv.prmtop CAGE_solv.inpcrd\n")
                f.write("quit\n")

            self._debug_print(f"  tleap input file written successfully")
        except Exception as e:
            self._debug_print(f"ERROR writing tleap input: {type(e).__name__}: {e}")
            raise

        print(f"\nGenerated tleap input: {tleap_path}")
        print(f"  Components: {', '.join(components)}")
        print(f"  Solvent: {self.solvent} ({solvent_box})")
        self._debug_print("generate_tleap_input() complete")

        return str(tleap_path)

    def run_tleap(self, input_file: Optional[str] = None):
        """
        Execute tleap with the generated input file.

        Parameters
        ----------
        input_file : str, optional
            Path to tleap input file. Uses input_tleap.in if not specified.
        """
        self._debug_print("="*60)
        self._debug_print("Starting run_tleap()")
        self._debug_print("="*60)
        self._debug_print(f"  input_file arg: {input_file}")

        print("\n" + "="*70)
        print("STEP 5: RUNNING TLEAP")
        print("="*70)

        input_file = input_file or str(self.working_dir / "input_tleap.in")
        self._debug_print(f"  Resolved input_file: {input_file}")
        self._debug_print(f"  File exists: {Path(input_file).exists()}")

        # Change to working directory
        original_dir = os.getcwd()
        self._debug_print(f"  Original directory: {original_dir}")
        self._debug_print(f"  Changing to working directory: {self.working_dir}")
        os.chdir(self.working_dir)

        # List files in working directory for debugging
        self._debug_print("  Files in working directory:")
        try:
            for f in sorted(self.working_dir.iterdir()):
                self._debug_print(f"    {f.name}", level=1)
        except Exception as e:
            self._debug_print(f"    ERROR listing directory: {e}", level=1)

        try:
            tleap_cmd = ['tleap', '-s', '-f', Path(input_file).name]
            self._debug_print(f"Executing command: {' '.join(tleap_cmd)}")
            print(f"\nExecuting: tleap -s -f {Path(input_file).name}")

            # Check if tleap is available
            import shutil
            tleap_path = shutil.which('tleap')
            self._debug_print(f"  tleap executable found: {tleap_path}")
            if not tleap_path:
                self._debug_print("  WARNING: tleap not found in PATH!")

            try:
                result = subprocess.run(
                    tleap_cmd,
                    capture_output=True,
                    text=True
                )
                self._debug_print(f"  subprocess completed with return code: {result.returncode}")
            except FileNotFoundError as e:
                self._debug_print(f"ERROR: tleap command not found: {e}")
                raise
            except Exception as e:
                self._debug_print(f"ERROR running tleap: {type(e).__name__}: {e}")
                raise

            # Print output
            if result.stdout:
                self._debug_print(f"  stdout length: {len(result.stdout)} chars")
                print("\n--- tleap output ---")
                print(result.stdout)

            if result.stderr:
                self._debug_print(f"  stderr length: {len(result.stderr)} chars")
                self._debug_print(f"  stderr content: {result.stderr[:500]}...")
                print("\n--- tleap errors ---")
                print(result.stderr)

            if result.returncode == 0:
                self._debug_print("  tleap completed successfully!")
                print("\ntleap completed successfully!")
                print("Generated files:")
                expected_files = ['ALL_dry.pdb', 'ALL_dry.prmtop', 'ALL_dry.inpcrd',
                          'CAGE_solv.pdb', 'CAGE_solv.prmtop', 'CAGE_solv.inpcrd']
                for f in expected_files:
                    file_path = self.working_dir / f
                    exists = file_path.exists()
                    self._debug_print(f"    {f}: exists={exists}")
                    if exists:
                        print(f"  - {f}")
                    else:
                        self._debug_print(f"    WARNING: Expected file {f} was not created!")
            else:
                self._debug_print(f"ERROR: tleap failed with return code: {result.returncode}")
                print(f"\ntleap failed with return code: {result.returncode}")

            self._debug_print("run_tleap() complete")

        finally:
            self._debug_print(f"Returning to original directory: {original_dir}")
            os.chdir(original_dir)

    # =========================================================================
    # Main workflow method
    # =========================================================================

    def run_all(self, skip_topology: bool = False):
        """
        Run the complete AutoBind workflow.

        This method executes all steps in sequence:
        1. Copy data files to working directory
        2. Prepare topology using metallicious (unless skip_topology=True)
        3. Extract frcmod parameters
        4. Position counterions and substrate
        5. Generate tleap input
        6. Run tleap

        Parameters
        ----------
        skip_topology : bool, optional
            If True, skip the metallicious topology preparation step.
            Useful if you already have cage_out.prmtop and cage_out.inpcrd.
        """
        self._debug_print("="*70)
        self._debug_print("STARTING run_all() - MAIN AUTOBIND WORKFLOW")
        self._debug_print("="*70)
        self._debug_print(f"  skip_topology: {skip_topology}")
        self._debug_print(f"  debug mode: {self.debug}")

        print("="*70)
        print("AUTOBIND: AUTOMATED AMBER TOPOLOGY GENERATION")
        print("="*70)
        print(f"\nInput PDB: {self.input_pdb}")
        print(f"Metal charges: {self.metal_charges}")
        print(f"Counterion type: {self.counterion_type} -> {self.counterions}")
        print(f"Substrate: {self.substrate_name} -> {self.substrate}")
        print(f"Solvent: {self.solvent_name} -> {self.solvent}")
        print(f"Working directory: {self.working_dir}")
        print(f"Parameter set: {self.parameter_set}")

        # Copy data files
        self._debug_print("\n" + "="*50)
        self._debug_print("WORKFLOW STEP: Copying data files")
        self._debug_print("="*50)
        try:
            self._copy_data_files()
            self._debug_print("Data files copied successfully")
        except Exception as e:
            self._debug_print(f"FATAL ERROR copying data files: {type(e).__name__}: {e}")
            raise

        # Step 1: Prepare topology
        self._debug_print("\n" + "="*50)
        self._debug_print("WORKFLOW STEP 1: Prepare topology")
        self._debug_print("="*50)
        if not skip_topology:
            try:
                self.prepare_topology()
                self._debug_print("Topology preparation completed successfully")
            except Exception as e:
                self._debug_print(f"FATAL ERROR in prepare_topology: {type(e).__name__}: {e}")
                raise
        else:
            self._debug_print("Skipping topology preparation (skip_topology=True)")
            print("\n[Skipping topology preparation - using existing files]")
            self.cage_prmtop = self.working_dir / 'cage_out.prmtop'
            self.cage_inpcrd = self.working_dir / 'cage_out.inpcrd'
            self._debug_print(f"  Using existing cage_prmtop: {self.cage_prmtop}")
            self._debug_print(f"    Exists: {self.cage_prmtop.exists()}")
            self._debug_print(f"  Using existing cage_inpcrd: {self.cage_inpcrd}")
            self._debug_print(f"    Exists: {self.cage_inpcrd.exists()}")

        # Step 2: Extract frcmod
        self._debug_print("\n" + "="*50)
        self._debug_print("WORKFLOW STEP 2: Extract frcmod")
        self._debug_print("="*50)
        try:
            self.extract_frcmod()
            self._debug_print("frcmod extraction completed successfully")
        except Exception as e:
            self._debug_print(f"FATAL ERROR in extract_frcmod: {type(e).__name__}: {e}")
            raise

        # Step 3: Position molecules
        self._debug_print("\n" + "="*50)
        self._debug_print("WORKFLOW STEP 3: Position molecules")
        self._debug_print("="*50)
        try:
            self.position_molecules()
            self._debug_print("Molecule positioning completed successfully")
        except Exception as e:
            self._debug_print(f"FATAL ERROR in position_molecules: {type(e).__name__}: {e}")
            raise

        # Step 4: Generate tleap input
        self._debug_print("\n" + "="*50)
        self._debug_print("WORKFLOW STEP 4: Generate tleap input")
        self._debug_print("="*50)
        try:
            self.generate_tleap_input()
            self._debug_print("tleap input generation completed successfully")
        except Exception as e:
            self._debug_print(f"FATAL ERROR in generate_tleap_input: {type(e).__name__}: {e}")
            raise

        # Step 5: Run tleap
        self._debug_print("\n" + "="*50)
        self._debug_print("WORKFLOW STEP 5: Run tleap")
        self._debug_print("="*50)
        try:
            self.run_tleap()
            self._debug_print("tleap execution completed")
        except Exception as e:
            self._debug_print(f"FATAL ERROR in run_tleap: {type(e).__name__}: {e}")
            raise

        self._debug_print("\n" + "="*70)
        self._debug_print("run_all() WORKFLOW COMPLETE")
        self._debug_print("="*70)

        print("\n" + "="*70)
        print("AUTOBIND WORKFLOW COMPLETE!")
        print("="*70)
        print(f"\nOutput files in: {self.working_dir}")
        print("\nKey output files:")
        print("  - CAGE_solv.prmtop: Solvated system topology")
        print("  - CAGE_solv.inpcrd: Solvated system coordinates")
        print("  - ALL_dry.prmtop: Dry system topology")
        print("  - ALL_dry.inpcrd: Dry system coordinates")

        # Final verification of output files
        self._debug_print("\nFinal output file verification:")
        output_files = ['CAGE_solv.prmtop', 'CAGE_solv.inpcrd', 'ALL_dry.prmtop', 'ALL_dry.inpcrd']
        for f in output_files:
            fpath = self.working_dir / f
            self._debug_print(f"  {f}: exists={fpath.exists()}")

    @classmethod
    def add_counterion_residue(cls, name: str, mol2_file: str, frcmod_file: str):
        """
        Add a new counterion residue to the available counterions.

        Parameters
        ----------
        name : str
            Residue name identifier for the counterion.
        mol2_file : str
            Filename of the mol2 file (should be in data/counterions).
        frcmod_file : str
            Filename of the frcmod file (should be in data/counterions).
        """
        cls.AVAILABLE_COUNTERIONS[name] = {'mol2': mol2_file, 'frcmod': frcmod_file}

    @classmethod
    def register_counterion_type(cls, type_name: str, residues: List[str],
                                  full_name: Optional[str] = None,
                                  charge: int = -1,
                                  description: Optional[str] = None):
        """
        Register a new counterion type in the library.

        This allows users to specify counterions by chemical name (e.g., 'BArF')
        instead of individual residue names.

        Parameters
        ----------
        type_name : str
            Chemical name for this counterion type (e.g., 'BArF', 'PF6').
        residues : list of str
            List of residue names that make up this counterion type.
            These must already be registered in AVAILABLE_COUNTERIONS.
        full_name : str, optional
            Full chemical name for documentation.
        charge : int, optional
            Charge of each counterion (default: -1).
        description : str, optional
            Description of this counterion type.

        Example
        -------
        >>> # Register a new counterion type
        >>> AutoBind.add_counterion_residue('PF6', 'PF6.mol2', 'PF6.frcmod')
        >>> AutoBind.register_counterion_type(
        ...     'PF6',
        ...     residues=['PF6', 'PF6', 'PF6', 'PF6'],
        ...     full_name='Hexafluorophosphate',
        ...     charge=-1
        ... )
        """
        entry = {'residues': residues, 'charge': charge}
        if full_name:
            entry['full_name'] = full_name
        if description:
            entry['description'] = description
        cls.COUNTERION_LIBRARY[type_name] = entry

    @classmethod
    def add_substrate_residue(cls, name: str, mol2_file: str, frcmod_file: str,
                               pdb_file: Optional[str] = None):
        """
        Add a new substrate residue to the available substrates.

        Parameters
        ----------
        name : str
            Residue name identifier for the substrate.
        mol2_file : str
            Filename of the mol2 file (should be in data/binding_substrates).
        frcmod_file : str
            Filename of the frcmod file (should be in data/binding_substrates).
        pdb_file : str, optional
            Filename of the pdb file (should be in data/binding_substrates).
        """
        substrate_info = {'mol2': mol2_file, 'frcmod': frcmod_file}
        if pdb_file:
            substrate_info['pdb'] = pdb_file
        cls.AVAILABLE_SUBSTRATES[name] = substrate_info

    @classmethod
    def register_substrate(cls, chemical_name: str, residue: str,
                            full_name: Optional[str] = None,
                            description: Optional[str] = None,
                            aliases: Optional[List[str]] = None):
        """
        Register a new substrate in the library.

        This allows users to specify substrates by chemical name instead of
        internal residue names.

        Parameters
        ----------
        chemical_name : str
            Chemical name for this substrate (e.g., 'pToluquinone').
        residue : str
            Internal residue name. Must be registered in AVAILABLE_SUBSTRATES.
        full_name : str, optional
            Full chemical name for documentation.
        description : str, optional
            Description of this substrate.
        aliases : list of str, optional
            Alternative names that should also map to this substrate.

        Example
        -------
        >>> AutoBind.add_substrate_residue('BZQ', 'BZQ.mol2', 'BZQ.frcmod')
        >>> AutoBind.register_substrate(
        ...     'benzoquinone',
        ...     residue='BZQ',
        ...     full_name='1,4-Benzoquinone',
        ...     aliases=['BQ', 'p-benzoquinone']
        ... )
        """
        entry = {'residue': residue}
        if full_name:
            entry['full_name'] = full_name
        if description:
            entry['description'] = description
        cls.SUBSTRATE_LIBRARY[chemical_name] = entry

        # Add aliases
        if aliases:
            for alias in aliases:
                cls.SUBSTRATE_LIBRARY[alias] = {'residue': residue}

    @classmethod
    def add_solvent(cls, name: str, lib_file: str, box_name: str,
                    full_name: Optional[str] = None,
                    aliases: Optional[List[str]] = None):
        """
        Add a new solvent to the available solvents.

        Parameters
        ----------
        name : str
            Name identifier for the solvent.
        lib_file : str
            Filename of the lib file (should be in data/solvent_box_info).
        box_name : str
            Name of the solvent box in tleap.
        full_name : str, optional
            Full chemical name for documentation.
        aliases : list of str, optional
            Alternative names (e.g., 'CH2Cl2' for 'DCM').
        """
        cls.AVAILABLE_SOLVENTS[name] = {'lib': lib_file, 'box_name': box_name}

        # Add to library with full name
        entry = {'identifier': name}
        if full_name:
            entry['full_name'] = full_name
        cls.SOLVENT_LIBRARY[name] = entry

        # Add aliases
        if aliases:
            for alias in aliases:
                cls.SOLVENT_LIBRARY[alias] = {'identifier': name}

    @classmethod
    def list_available(cls):
        """Print all available counterion types, substrates, and solvents."""
        print("="*60)
        print("AVAILABLE COMPONENTS")
        print("="*60)

        print("\nCounterion Types:")
        for name, info in cls.COUNTERION_LIBRARY.items():
            full = info.get('full_name', '')
            residues = info.get('residues', [])
            print(f"  {name}: {full}")
            print(f"    Residues: {residues}")

        print("\nSubstrates:")
        for name, info in cls.SUBSTRATE_LIBRARY.items():
            if 'full_name' in info:  # Only show primary entries, not aliases
                full = info.get('full_name', '')
                residue = info.get('residue', '')
                print(f"  {name}: {full} -> {residue}")

        print("\nSolvents:")
        for name, info in cls.SOLVENT_LIBRARY.items():
            if 'full_name' in info:  # Only show primary entries, not aliases
                full = info.get('full_name', '')
                identifier = info.get('identifier', '')
                print(f"  {name}: {full} -> {identifier}")


# Convenience function for quick usage
def run_autobind(
    input_pdb: str,
    metal_charges: Optional[Dict[str, int]] = None,
    counterion_type: str = 'BArF',
    substrate: str = 'pToluquinone',
    solvent: str = 'DCM',
    working_dir: Optional[str] = None,
    debug: bool = False,
    **kwargs
) -> AutoBind:
    """
    Convenience function to run the complete AutoBind workflow.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    metal_charges : dict, optional
        Metal charges. Default is {'Pd': 2}.
    counterion_type : str, optional
        Counterion type name. Default is 'BArF'.
        Available: 'BArF' (automatically uses 4 BArF anions)
    substrate : str, optional
        Substrate name. Default is 'pToluquinone'.
        Available: 'pToluquinone', 'PTQ', 'toluquinone'
    solvent : str, optional
        Solvent name. Default is 'DCM'.
        Available: 'DCM', 'dichloromethane', 'CH2Cl2'
    working_dir : str, optional
        Working directory for output files.
    debug : bool, optional
        Enable verbose debug output. Default is False.
    **kwargs
        Additional keyword arguments passed to AutoBind.

    Returns
    -------
    AutoBind
        The AutoBind instance after running the workflow.

    Example
    -------
    >>> from autobind import run_autobind
    >>> ab = run_autobind("my_cage.pdb")  # Uses BArF and pToluquinone by default
    >>> ab = run_autobind("my_cage.pdb", debug=True)  # With debug output
    >>> ab = run_autobind("my_cage.pdb", counterion_type='BArF', substrate='pToluquinone')
    """
    ab = AutoBind(
        input_pdb=input_pdb,
        metal_charges=metal_charges,
        counterion_type=counterion_type,
        substrate=substrate,
        solvent=solvent,
        working_dir=working_dir,
        debug=debug,
        **kwargs
    )
    ab.run_all()
    return ab


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="AutoBind: Automated AMBER topology generation for cage structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults (BArF counterions, pToluquinone substrate)
  python -m autobind.autobind my_cage.pdb

  # Specify counterion and substrate by chemical name
  python -m autobind.autobind my_cage.pdb --counterion BArF --substrate pToluquinone

  # Without substrate
  python -m autobind.autobind my_cage.pdb --no-substrate

Available counterion types: BArF, Cl (chloride - uses tleap built-in)
Available substrates: pToluquinone, PTQ, toluquinone
Available solvents: DCM, ACE (acetone), DMSO, HCN (acetonitrile), Nitro (nitromethane), THF, oDFB, OPC (water), TIP3P (water)
        """
    )
    parser.add_argument("input_pdb", help="Input PDB file with cage structure")
    parser.add_argument("--metal", "-m", nargs=2, action="append", metavar=("SYMBOL", "CHARGE"),
                        help="Metal symbol and charge (can be used multiple times)")
    parser.add_argument("--counterion", "-c", default='BArF',
                        help="Counterion type (default: BArF)")
    parser.add_argument("--substrate", "-s", default="pToluquinone",
                        help="Substrate name (default: pToluquinone)")
    parser.add_argument("--no-substrate", action="store_true",
                        help="Don't include a substrate")
    parser.add_argument("--no-counterions", action="store_true",
                        help="Don't include counterions")
    parser.add_argument("--solvent", default="DCM", help="Solvent name (default: DCM)")
    parser.add_argument("--working-dir", "-w", help="Working directory")
    parser.add_argument("--parameter-set", "-p", help="Path to parameter set file")
    parser.add_argument("--skip-topology", action="store_true",
                        help="Skip metallicious topology preparation")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable verbose debug output for troubleshooting")

    args = parser.parse_args()

    # Parse metal charges
    metal_charges = {}
    if args.metal:
        for symbol, charge in args.metal:
            metal_charges[symbol] = int(charge)
    else:
        metal_charges = {'Pd': 2}

    ab = AutoBind(
        input_pdb=args.input_pdb,
        metal_charges=metal_charges,
        counterion_type=args.counterion,
        substrate=args.substrate,
        solvent=args.solvent,
        working_dir=args.working_dir,
        parameter_set=args.parameter_set,
        process_counterions=not args.no_counterions,
        process_substrate=not args.no_substrate,
        debug=args.debug
    )
    ab.run_all(skip_topology=args.skip_topology)
