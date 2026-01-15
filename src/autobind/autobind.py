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
    """

    # =========================================================================
    # CHEMICAL LIBRARIES - Maps common names to internal file identifiers
    # =========================================================================

    # Counterion type library: maps chemical names to list of residue names
    # Each counterion type can have multiple residues (e.g., 4 BArF anions)
    COUNTERION_LIBRARY = {
        'BArF': {
            'full_name': 'Tetrakis(3,5-bis(trifluoromethyl)phenyl)borate',
            'residues': ['BFV', 'BFW', 'BFX', 'BFY'],
            'charge': -1,
            'description': 'Weakly coordinating anion, 4 copies for Pd2L4 cage'
        },
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
    AVAILABLE_SOLVENTS = {
        'DCM': {'lib': 'DCMequilbox.lib', 'box_name': 'DCMequilbox'},
        'ACE': {'lib': 'ACEequilbox.lib', 'box_name': 'ACEequilbox'},
        'DMSO': {'lib': 'DMSOequilbox.lib', 'box_name': 'DMSOequilbox'},
        'HCN': {'lib': 'HCNequilbox.lib', 'box_name': 'HCNequilbox'},
        'Nitro': {'lib': 'Nitroequilbox.lib', 'box_name': 'Nitroequilbox'},
        'THF': {'lib': 'THFequilbox.lib', 'box_name': 'THFequilbox'},
        'oDFB': {'lib': 'oDFBequilbox.lib', 'box_name': 'oDFBequilbox'},
    }

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
        lj_type: str = 'merz-opc'
    ):
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
        self._validate_inputs()

        # Generated file paths (will be set during processing)
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
        """
        if not self.process_counterions:
            return []

        # Check if it's a known counterion type
        if counterion_type in self.COUNTERION_LIBRARY:
            residues = self.COUNTERION_LIBRARY[counterion_type]['residues']
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
        # Check input PDB exists
        if not Path(self.input_pdb).exists():
            raise FileNotFoundError(f"Input PDB file not found: {self.input_pdb}")

        # Check counterions exist in data folder
        if self.process_counterions:
            for ci in self.counterions:
                if ci not in self.AVAILABLE_COUNTERIONS:
                    raise ValueError(f"Counterion residue '{ci}' not found. Available: {list(self.AVAILABLE_COUNTERIONS.keys())}")
                mol2_path = self.counterions_dir / self.AVAILABLE_COUNTERIONS[ci]['mol2']
                if not mol2_path.exists():
                    raise FileNotFoundError(f"Counterion mol2 file not found: {mol2_path}")

        # Check substrate exists
        if self.process_substrate and self.substrate:
            if self.substrate not in self.AVAILABLE_SUBSTRATES:
                raise ValueError(f"Substrate residue '{self.substrate}' not found. Available: {list(self.AVAILABLE_SUBSTRATES.keys())}")
            mol2_path = self.substrates_dir / self.AVAILABLE_SUBSTRATES[self.substrate]['mol2']
            if not mol2_path.exists():
                raise FileNotFoundError(f"Substrate mol2 file not found: {mol2_path}")

        # Check solvent exists
        if self.solvent:
            if self.solvent not in self.AVAILABLE_SOLVENTS:
                raise ValueError(f"Solvent '{self.solvent}' not found. Available: {list(self.AVAILABLE_SOLVENTS.keys())}")

    def _copy_data_files(self):
        """Copy required data files to working directory."""
        print("\nCopying data files to working directory...")

        # Copy counterion files
        if self.process_counterions:
            for ci in self.counterions:
                for file_type in ['mol2', 'frcmod']:
                    src = self.counterions_dir / self.AVAILABLE_COUNTERIONS[ci][file_type]
                    dst = self.working_dir / self.AVAILABLE_COUNTERIONS[ci][file_type]
                    if src.exists():
                        shutil.copy2(src, dst)
                        print(f"  Copied {src.name}")

        # Copy substrate files
        if self.process_substrate and self.substrate:
            for file_type in ['mol2', 'frcmod', 'pdb']:
                if file_type in self.AVAILABLE_SUBSTRATES[self.substrate]:
                    src = self.substrates_dir / self.AVAILABLE_SUBSTRATES[self.substrate][file_type]
                    dst = self.working_dir / self.AVAILABLE_SUBSTRATES[self.substrate][file_type]
                    if src.exists():
                        shutil.copy2(src, dst)
                        print(f"  Copied {src.name}")

        # Copy solvent files
        if self.solvent:
            solvent_info = self.AVAILABLE_SOLVENTS[self.solvent]
            src = self.solvent_dir / solvent_info['lib']
            dst = self.working_dir / solvent_info['lib']
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied {src.name}")

            # Copy leaprc.gaff2
            leaprc_src = self.solvent_dir / "leaprc.gaff2"
            leaprc_dst = self.working_dir / "leaprc.gaff2"
            if leaprc_src.exists():
                shutil.copy2(leaprc_src, leaprc_dst)
                print(f"  Copied leaprc.gaff2")

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
        if supramolecular_structure is None:
            raise ImportError("metallicious is required for topology preparation. Install with: pip install metallicious")
        if pmd is None:
            raise ImportError("parmed is required. Install with: pip install parmed")

        print("="*70)
        print("STEP 1: PREPARING TOPOLOGY (metallicious)")
        print("="*70)

        # Change to working directory
        original_dir = os.getcwd()
        os.chdir(self.working_dir)

        try:
            print(f"\nLoading cage structure from {self.input_pdb}...")
            print(f"Metal charges: {self.metal_charges}")
            print(f"LJ type: {self.lj_type}")

            cage = supramolecular_structure(
                str(Path(original_dir) / self.input_pdb) if not Path(self.input_pdb).is_absolute() else self.input_pdb,
                metal_charges=self.metal_charges,
                LJ_type=self.lj_type
            )
            cage.prepare_initial_topology()
            cage.parametrize(out_coord='cage_out.pdb', out_topol='cage_out.top')

            # Store paths
            self.cage_pdb = self.working_dir / 'cage_out.pdb'
            self.cage_top = self.working_dir / 'cage_out.top'

            # Load GROMACS topology and structure
            print("\nLoading GROMACS topology...")
            gro_top = pmd.load_file('cage_out.top', xyz='cage_out.pdb')

            # Save as Amber format
            print("Saving Amber topology and coordinates...")
            gro_top.save('cage_out.prmtop', format='amber')
            gro_top.save('cage_out.inpcrd', format='rst7')

            self.cage_prmtop = self.working_dir / 'cage_out.prmtop'
            self.cage_inpcrd = self.working_dir / 'cage_out.inpcrd'

            print("\nTopology preparation complete!")
            print(f"  Created: {self.cage_prmtop.name}")
            print(f"  Created: {self.cage_inpcrd.name}")

        finally:
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
        selector: str = ":*"
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
        """
        if pmd is None or AmberMask is None:
            raise ImportError("parmed is required. Install with: pip install parmed")

        print("\n" + "="*70)
        print("STEP 2: EXTRACTING FORCE FIELD PARAMETERS")
        print("="*70)

        # Use default paths if not specified
        prmtop = prmtop or str(self.working_dir / "cage_out.prmtop")
        inpcrd = inpcrd or str(self.working_dir / "cage_out.inpcrd")

        print(f"\nInput prmtop: {prmtop}")
        print(f"Input inpcrd: {inpcrd}")
        print(f"Output mol2: {out_mol2}")
        print(f"Output frcmod: {out_frcmod}")
        print(f"Selector: {selector}")

        # Change to working directory
        original_dir = os.getcwd()
        os.chdir(self.working_dir)

        try:
            struct = pmd.load_file(prmtop, inpcrd)
            struct.load_atom_info()

            # Identify special types
            special_types, special_atoms = self._collect_special_types(struct, selector)
            print(f"\nSelector {selector!r}: selected {len(special_atoms)} atoms, {len(special_types)} types")

            # Save subset as mol2
            sel_idx = sorted(a.idx for a in special_atoms)
            sub = struct[sel_idx]
            sub.save(out_mol2, format="mol2", overwrite=True)
            out_pdb = out_mol2.rsplit(".", 1)[0] + ".pdb"
            sub.save(out_pdb, format="pdb", overwrite=True)

            # Reload structure
            struct = pmd.load_file(prmtop, inpcrd)

            if not special_types:
                raise RuntimeError("No special types identified.")

            def involves_special(types):
                return any(t in special_types for t in types)

            # Collect parameters
            type_to_mass = {}
            for a in struct.atoms:
                if a.type in special_types and a.type not in type_to_mass:
                    type_to_mass[a.type] = a.mass

            bonds = {}
            angles = {}
            diheds = {}
            improps = {}

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

            # Dihedrals
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

            # Nonbonded parameters
            nonbon = {}
            for a in struct.atoms:
                t = a.type
                if t in special_types and a.atom_type is not None:
                    sigma = getattr(a.atom_type, "sigma", None)
                    eps = getattr(a.atom_type, "epsilon", None)
                    if sigma is None or eps is None:
                        rmin = getattr(a.atom_type, "rmin", None)
                        eps2 = getattr(a.atom_type, "epsilon", None)
                        if rmin is None or eps2 is None:
                            continue
                        sigma = rmin * (2.0 ** (-1.0/6.0))
                        eps = eps2
                    nonbon[t] = (sigma, eps)

            # Write frcmod
            self._write_frcmod(out_frcmod, type_to_mass, bonds, angles, diheds, improps, nonbon)

            self.cage_mol2 = self.working_dir / out_mol2
            self.cage_frcmod = self.working_dir / out_frcmod

            print(f"\nWrote: {out_mol2}")
            print(f"Wrote: {out_frcmod}")
            print(f"Special types: {' '.join(sorted(special_types))}")

        finally:
            os.chdir(original_dir)

    def _collect_special_types(self, struct, selector: str) -> Tuple[Set[str], Set[Any]]:
        """Collect atom types for the given selector mask."""
        selector = selector.strip()

        mask = AmberMask(struct, selector)
        indices = mask.Selected()

        if not indices:
            raise RuntimeError(f"Mask '{selector}' selected 0 atoms. Check residue names.")

        special_atoms = {struct.atoms[i] for i in indices}
        special_types = {a.type for a in special_atoms if a.type is not None}

        if not special_types:
            raise RuntimeError(f"Mask '{selector}' selected atoms but none had atom types.")

        return special_types, special_atoms

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
            for t, (sigma, eps) in sorted(nonbon.items()):
                f.write(f"{t:<6} {self._fmt(sigma)}  {self._fmt(eps)}\n")

            f.write("\nEND\n")

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
        print("\n" + "="*70)
        print("STEP 3: POSITIONING MOLECULES")
        print("="*70)

        # Use default paths if not specified
        if input_pdb is None:
            input_pdb = str(self.working_dir / "CAGE.pdb")

        if output_pdb is None:
            base_name = Path(input_pdb).stem
            output_pdb = f"{base_name}_positioned.pdb"

        # Change to working directory
        original_dir = os.getcwd()
        os.chdir(self.working_dir)

        try:
            print(f"\nReading {input_pdb}...")
            pdb_residues = self._read_pdb(input_pdb)
            print(f"  Found {len(pdb_residues)} residues")

            # Calculate cage bounds
            print("\nAnalyzing cage structure...")
            cage_center, min_bounds, max_bounds, max_radius, pd1_coords, pd2_coords, linker_coords = self._calculate_cage_bounds(pdb_residues)
            print(f"  Cage center: ({cage_center[0]:.3f}, {cage_center[1]:.3f}, {cage_center[2]:.3f})")
            print(f"  Cage radius: {max_radius:.3f} A")

            # Process counterions
            counterion_data = None
            counterion_resids = None
            if self.process_counterions and self.counterions:
                counterion_data, counterion_resids = self._process_counterions(
                    cage_center, max_radius,
                    counterion_names=self.counterions,
                    new_resids=self._counterion_resids,
                    placement_multiplier=self.counterion_placement_multiplier
                )

            # Process substrate
            substrate_atoms = None
            if self.process_substrate and self.substrate:
                substrate_atoms, _, _, _ = self._process_substrate(
                    cage_center, pd1_coords, pd2_coords, linker_coords,
                    substrate_name=self.substrate,
                    substrate_resid=self._substrate_resid,
                    covalent_cutoff=self.substrate_covalent_cutoff
                )

            # Build complete structure
            complete_pdb_residues = self._build_complete_structure(
                pdb_residues,
                counterion_data=counterion_data,
                counterion_resids=counterion_resids,
                substrate_atoms=substrate_atoms,
                substrate_name=self.substrate,
                substrate_resid=self._substrate_resid
            )

            # Write complete PDB file
            print(f"\nWriting complete PDB file to {output_pdb}...")
            self._write_pdb(output_pdb, complete_pdb_residues)
            self.final_pdb = self.working_dir / output_pdb
            print(f"  Updated {output_pdb}")

            # Print summary
            self._print_summary(complete_pdb_residues)

        finally:
            os.chdir(original_dir)

    def _read_pdb(self, filename: str) -> Dict[int, List[Atom]]:
        """Read PDB file and return atoms grouped by residue."""
        residues = {}
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('HETATM') or line.startswith('ATOM'):
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

    def _read_mol2(self, filename: str) -> Tuple[List[Mol2Atom], List[str], Dict, List[str]]:
        """Read mol2 file and return atoms and bonds."""
        atoms = []
        bonds = []
        molecule_info = {}

        with open(filename, 'r') as f:
            lines = f.readlines()

        in_molecule = False
        in_atom = False
        in_bond = False

        for i, line in enumerate(lines):
            if '@<TRIPOS>MOLECULE' in line:
                in_molecule = True
                in_atom = False
                in_bond = False
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
                    atom = Mol2Atom(
                        int(parts[0]), parts[1],
                        float(parts[2]), float(parts[3]), float(parts[4]),
                        parts[5], int(parts[6]), parts[7], float(parts[8])
                    )
                    atoms.append(atom)

            elif in_bond and line.strip():
                bonds.append(line)

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
        # Get Pd positions (residues 1 and 2)
        pd1_coords = residues[1][0].coords()
        pd2_coords = residues[2][0].coords()

        # Get all linker atoms (residues 3-6)
        linker_coords = []
        for resid in [3, 4, 5, 6]:
            if resid in residues:
                for atom in residues[resid]:
                    linker_coords.append(atom.coords())

        linker_coords = np.array(linker_coords)

        # Calculate cage center
        cage_center = (pd1_coords + pd2_coords) / 2

        # Calculate cage bounds
        all_cage_coords = np.vstack([linker_coords, [pd1_coords, pd2_coords]])
        min_bounds = np.min(all_cage_coords, axis=0)
        max_bounds = np.max(all_cage_coords, axis=0)

        # Calculate max distance from center
        distances = np.linalg.norm(all_cage_coords - cage_center, axis=1)
        max_radius = np.max(distances)

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
        print("\n" + "-"*50)
        print("COUNTERION POSITIONING")
        print("-"*50)

        print("\n1. Reading counterion mol2 files...")
        counterion_data = {}

        for name in counterion_names:
            atoms, bonds, mol_info, orig_lines = self._read_mol2(f'{name}.mol2')
            counterion_data[name] = {
                'atoms': atoms,
                'bonds': bonds,
                'mol_info': mol_info,
                'orig_lines': orig_lines
            }
            print(f"   {name}: {len(atoms)} atoms")

        print(f"\n2. Positioning counterions (distance = {placement_multiplier}x radius)...")
        counterion_atoms_list = [counterion_data[name]['atoms'] for name in counterion_names]
        positioned_counterions = self._position_counterions_outside(
            cage_center, max_radius, counterion_atoms_list, placement_multiplier
        )

        print("\n3. Updating counterion residue numbers...")
        for i, (name, new_resid) in enumerate(zip(counterion_names, new_resids)):
            self._update_residue_numbers(positioned_counterions[i], new_resid, name)
            counterion_data[name]['atoms'] = positioned_counterions[i]
            print(f"   {name} -> Residue {new_resid}")

        print("\n4. Writing updated counterion mol2 files...")
        for name in counterion_names:
            self._write_mol2(
                f'{name}.mol2',
                counterion_data[name]['atoms'],
                counterion_data[name]['bonds'],
                counterion_data[name]['mol_info'],
                counterion_data[name]['orig_lines']
            )
            print(f"   Updated {name}.mol2")

        return counterion_data, new_resids

    def _process_substrate(self, cage_center: np.ndarray, pd1_coords: np.ndarray,
                           pd2_coords: np.ndarray, linker_coords: np.ndarray,
                           substrate_name: str, substrate_resid: int,
                           covalent_cutoff: float = 2.5) -> Tuple:
        """Position substrate inside the cage."""
        print("\n" + "-"*50)
        print("SUBSTRATE POSITIONING")
        print("-"*50)

        print(f"\n1. Reading {substrate_name} mol2 file...")
        substrate_atoms, substrate_bonds, substrate_mol_info, substrate_orig_lines = self._read_mol2(f'{substrate_name}.mol2')
        print(f"   {substrate_name}: {len(substrate_atoms)} atoms")

        print(f"\n2. Positioning {substrate_name} inside cage...")
        positioned_substrate = self._position_substrate_inside(
            cage_center, pd1_coords, pd2_coords, linker_coords,
            substrate_atoms, covalent_cutoff=covalent_cutoff
        )

        print(f"\n3. Updating {substrate_name} residue number to {substrate_resid}...")
        self._update_residue_numbers(positioned_substrate, substrate_resid, substrate_name)

        print(f"\n4. Writing updated {substrate_name} mol2 file...")
        self._write_mol2(f'{substrate_name}.mol2', positioned_substrate,
                        substrate_bonds, substrate_mol_info, substrate_orig_lines)
        print(f"   Updated {substrate_name}.mol2")

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
        print("\n" + "="*70)
        print("STEP 4: GENERATING TLEAP INPUT")
        print("="*70)

        tleap_path = self.working_dir / output_file

        # Build component lists
        components = ['CAGE']
        mol2_loads = ['CAGE = loadmol2 CAGE.mol2']
        frcmod_loads = ['loadamberparams CAGE.frcmod']

        # Add substrate
        if self.process_substrate and self.substrate:
            components.append(self.substrate)
            mol2_loads.append(f'{self.substrate} = loadmol2 {self.substrate}.mol2')
            frcmod_loads.append(f'loadamberparams {self.substrate}.frcmod')

        # Add counterions
        if self.process_counterions:
            for ci in self.counterions:
                components.append(ci)
                mol2_loads.append(f'{ci} = loadmol2 {ci}.mol2')
                frcmod_loads.append(f'loadamberparams {ci}.frcmod')

        # Build residue combination list (cage residues + other components)
        # Cage residues are typically: PD1 PD2 LA1 LB1 LC1 LD1
        cage_residues = ['PD1', 'PD2', 'LA1', 'LB1', 'LC1', 'LD1']
        combine_list = cage_residues.copy()

        if self.process_substrate and self.substrate:
            combine_list.append(self.substrate)
        if self.process_counterions:
            combine_list.extend(self.counterions)

        # Solvent configuration
        solvent_info = self.AVAILABLE_SOLVENTS.get(self.solvent, {})
        solvent_lib = solvent_info.get('lib', 'DCMequilbox.lib')
        solvent_box = solvent_info.get('box_name', 'DCMequilbox')

        # Write tleap input
        with open(tleap_path, 'w') as f:
            f.write("# AutoBind generated tleap input\n")
            f.write("source leaprc.gaff2\n")
            f.write("source leaprc.water.opc\n\n")

            f.write("# Load mol2 files\n")
            for load in mol2_loads:
                f.write(f"{load}\n")

            f.write("\n# Load frcmod parameters\n")
            for load in frcmod_loads:
                f.write(f"{load}\n")

            f.write("\n# Combine all components\n")
            f.write(f"mol = combine {{ {' '.join(combine_list)} }}\n")
            f.write("check mol\n\n")

            f.write("# Save dry system\n")
            f.write("savepdb mol ALL_dry.pdb\n")
            f.write("saveamberparm mol ALL_dry.prmtop ALL_dry.inpcrd\n\n")

            f.write("# Solvate\n")
            f.write(f"loadoff {solvent_lib}\n")
            f.write(f"solvateBox mol {solvent_box} 12.0\n")
            f.write("addions mol Na+ 0\n")
            f.write("addions mol Cl- 0\n\n")

            f.write("# Save solvated system\n")
            f.write("savepdb mol CAGE_solv.pdb\n")
            f.write("saveamberparm mol CAGE_solv.prmtop CAGE_solv.inpcrd\n")
            f.write("quit\n")

        print(f"\nGenerated tleap input: {tleap_path}")
        print(f"  Components: {', '.join(components)}")
        print(f"  Solvent: {self.solvent} ({solvent_box})")

        return str(tleap_path)

    def run_tleap(self, input_file: Optional[str] = None):
        """
        Execute tleap with the generated input file.

        Parameters
        ----------
        input_file : str, optional
            Path to tleap input file. Uses input_tleap.in if not specified.
        """
        print("\n" + "="*70)
        print("STEP 5: RUNNING TLEAP")
        print("="*70)

        input_file = input_file or str(self.working_dir / "input_tleap.in")

        # Change to working directory
        original_dir = os.getcwd()
        os.chdir(self.working_dir)

        try:
            print(f"\nExecuting: tleap -s -f {Path(input_file).name}")

            result = subprocess.run(
                ['tleap', '-s', '-f', Path(input_file).name],
                capture_output=True,
                text=True
            )

            # Print output
            if result.stdout:
                print("\n--- tleap output ---")
                print(result.stdout)

            if result.stderr:
                print("\n--- tleap errors ---")
                print(result.stderr)

            if result.returncode == 0:
                print("\ntleap completed successfully!")
                print("Generated files:")
                for f in ['ALL_dry.pdb', 'ALL_dry.prmtop', 'ALL_dry.inpcrd',
                          'CAGE_solv.pdb', 'CAGE_solv.prmtop', 'CAGE_solv.inpcrd']:
                    if (self.working_dir / f).exists():
                        print(f"  - {f}")
            else:
                print(f"\ntleap failed with return code: {result.returncode}")

        finally:
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
        self._copy_data_files()

        # Step 1: Prepare topology
        if not skip_topology:
            self.prepare_topology()
        else:
            print("\n[Skipping topology preparation - using existing files]")
            self.cage_prmtop = self.working_dir / 'cage_out.prmtop'
            self.cage_inpcrd = self.working_dir / 'cage_out.inpcrd'

        # Step 2: Extract frcmod
        self.extract_frcmod()

        # Step 3: Position molecules
        self.position_molecules()

        # Step 4: Generate tleap input
        self.generate_tleap_input()

        # Step 5: Run tleap
        self.run_tleap()

        print("\n" + "="*70)
        print("AUTOBIND WORKFLOW COMPLETE!")
        print("="*70)
        print(f"\nOutput files in: {self.working_dir}")
        print("\nKey output files:")
        print("  - CAGE_solv.prmtop: Solvated system topology")
        print("  - CAGE_solv.inpcrd: Solvated system coordinates")
        print("  - ALL_dry.prmtop: Dry system topology")
        print("  - ALL_dry.inpcrd: Dry system coordinates")

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
    >>> ab = run_autobind("my_cage.pdb", counterion_type='BArF', substrate='pToluquinone')
    """
    ab = AutoBind(
        input_pdb=input_pdb,
        metal_charges=metal_charges,
        counterion_type=counterion_type,
        substrate=substrate,
        solvent=solvent,
        working_dir=working_dir,
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

Available counterion types: BArF
Available substrates: pToluquinone, PTQ, toluquinone
Available solvents: DCM, ACE (acetone), DMSO, HCN (acetonitrile), Nitro (nitromethane), THF, oDFB
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
        process_substrate=not args.no_substrate
    )
    ab.run_all(skip_topology=args.skip_topology)
