"""
OFF2.0 Calculator implementation
"""

from typing import Union

import openff.toolkit
import numpy as np
import pandas as pd
from rdkit import Chem

try:
    from openmm import unit
    from openmm import CustomTorsionForce
    from openmm import LangevinMiddleIntegrator, System
    from openmm.app import Simulation, PDBReporter, Topology

    from openff.toolkit.topology import Molecule
    from openff.toolkit.typing.engines.smirnoff import ForceField

    from openff.toolkit.utils import toolkit_registry
    found_off_deps = True

except ImportError:
    found_off_deps = False

from ..utils import bash
from typing import Iterable
from ..tools.torsion_profiler import _read_tp_pdb
from ..utils import store_mol_db, mols_to_moldf
from ..tools.torsion_profiler.utils import get_angles_of_torsion_profile_mol
from ._abstractcalculator import _AbstractCalculator


if found_off_deps:
    # Build ANI Class (from torch openMM)
    class OffCalculator(_AbstractCalculator):
        """
        Implementation of the OpenFF Calculator
        """
        # Coordinate Optimization:
        _optimize_structure: bool
        _optimize_structure_nsteps: int
        _optimize_structure_tol: float

        # Magics
        def __init__(
            self,
            ff_model: str = "openff-2.0.0.offxml",
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = 100000,
            optimize_structure_tol: float = 10**-6,
            optimize_structure_write_out: bool = True,
        ):
            """
            constructor of the openff calculator

            Parameters
            ----------
            ff_model: str
                rel path to the ff file.
            optimize_structure: bool
                execute optimization
            optimize_structure_nsteps: int
                number of optimization steps
            optimize_structure_tol: float
                convergence tolerance of optimization
            optimize_structure_write_out: bool
                write out files?
            """
            super().__init__()

            self.ff_model = ff_model
            self._optimize_structure = optimize_structure
            self._optimize_structure_nsteps = optimize_structure_nsteps
            self._optimize_structure_tol = optimize_structure_tol
            self.simulation = None
            self.omm_top = None
            self.omm_sys = None
            self.torsion_atom_ids = None
            self._additional_torsions = None

            # Functions
        ## Privates
        def __set_tors_restraints_for_torsion(
            self, torsion_atom_ids: tuple[int, int, int, int], angle: float, k: float = 1
        )->CustomTorsionForce:
            """
            build torsion restraints
            Parameters
            ----------
            torsion_atom_ids: tuple[int, int, int, int]
                atom ids defining the torsion
            angle: float
                target angle
            k: float
                force constant

            Returns
            -------
            CustomTorsionForce
                torsion restraint
            """
            restraint = CustomTorsionForce("0.5*k*(theta-theta0)^2")
            restraint.setName("TorsionRestraint")
            restraint.addGlobalParameter("k", k * unit.kilojoules_per_mole)
            restraint.addPerTorsionParameter("theta0")

            angle = np.deg2rad(angle) * unit.radians
            restraint.addTorsion(
                torsion_atom_ids[0],
                torsion_atom_ids[1],
                torsion_atom_ids[2],
                torsion_atom_ids[3],
                [angle],
            )

            return restraint

        def __build_dummy_openmm_simulation(self, omm_top: Topology, omm_sys: System) -> Simulation:
            """
                Construct an openMM simulation

            Parameters
            ----------
            omm_top: Topology
                topology
            omm_sys: System
                system

            Returns
            -------
            Simulation
                constructed system.

            """
            # Settings
            temperature = 298.15 * unit.kelvin
            friction_coeff = 1 / unit.picosecond
            time_step = 1 * unit.femtosecond

            integrator = LangevinMiddleIntegrator(temperature, friction_coeff, time_step)
            return Simulation(omm_top, omm_sys, integrator)

        def __calculate_openmm_optimized(self, off_mol: Molecule, simulation: Simulation,
                                         reporter: PDBReporter = None, nsteps: int = 100000,
                                         tol_emin: float = 10 ** -12 * unit.kilojoule_per_mole) -> dict[int, float]:
            """
                calculate optimization

            Parameters
            ----------
            off_mol:Molecule
                openff toolkit molecule
            simulation: Simulation
                simulation obj.
            reporter: PDBReporter
                write out the pdb here
            nsteps: int
                number of optimization steps
            tol_emin: float
                convergence limit

            Returns
            -------
            dict[int, float]
                potential energy per conformer
            """
            conf = off_mol.conformers[0].to_openmm()

            # do minimization and calc ani
            ## AddTors res
            if self.torsion_atom_ids is not None:
                i, j, k, l = self.torsion_atom_ids
                r = self.__set_tors_restraints_for_torsion(
                    torsion_atom_ids=self.torsion_atom_ids, angle=0, k=10**5
                )
                simulation.system.addForce(r)

            if self._additional_torsions is not None:
                for i, additional_torsion in enumerate(self._additional_torsions, start=1):
                    ai, aj, ak, al, angle2 = additional_torsion
                    rt = self.__set_tors_restraints_for_torsion(
                        torsion_atom_ids=(ai, aj, ak, al), angle=angle2, k=10**5
                    )
                    simulation.system.addForce(rt)
                    rt.setTorsionParameters(0, ai, aj, ak, al, [np.deg2rad(angle2)])

            self.simulation = simulation
            if self.torsion_atom_ids is not None:
                angle = np.round(
                    Chem.rdMolTransforms.GetDihedralDeg(
                        off_mol.to_rdkit().GetConformer(), *self.torsion_atom_ids
                    )
                )
                # swap angle
                if self.torsion_atom_ids is not None:
                    r.setTorsionParameters(0, i, j, k, l, [np.deg2rad(angle)])

            simulation.context.setPositions(conf)
            simulation.context.reinitialize()
            simulation.context.setPositions(conf)
            if self.torsion_atom_ids is not None:
                r.updateParametersInContext(simulation.context)

            ## set constraints on tors atoms:
            if self._optimize_structure:
                simulation.minimizeEnergy(tolerance=tol_emin, maxIterations=nsteps)

            cstate = simulation.context.getState(
                getPositions=True,
                getEnergy=True,
            )
            calc_v = cstate.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)

            if reporter is not None:
                reporter.report(simulation=simulation, state=cstate)

            res_off_mol = openff.toolkit.Molecule(off_mol)
            res_off_mol.conformers[0] = cstate.getPositions()
            rdmol= off_mol.to_rdkit()
            rdmol.SetProp("potential_energy", str(calc_v))
            return rdmol

        def __calculate_profile_with_openmm(
            self,
            off_mols: list[Molecule],
            torsion_atom_ids: tuple[int, int, int, int] = None,
            optimization_nsteps: int = 1000000,
            optimization_tol: float = 10**-12 * unit.kilojoule_per_mole,
            _conf_range: Iterable[int] = None,
            out_file_path: str = None,
            _additional_pos_res: list[int] = None,
            _additional_torsions: list[tuple[int, int, int, int, float]] = None,
        )->dict[int, float]:
            """
                calculate a full profile.

            Parameters
            ----------
            off_mol: Molecule
            torsion_atom_ids: tuple[int, int, int, int]
                atom ids defining the torsion
            write_optimized_coords: bool
                if true write out the coordinates
            optimization_nsteps: int
                number of optimization steps
            optimization_tol: float
                convergence tolerance.
            _conf_range: Iterable[int]
                conformers to be optimized, if none -> all
            out_file_path: str
                write out the result .tsv here.
            _additional_pos_res: list[int]
                additional constraints.
            _additional_torsions: list[tuple[int, int, int, int, float]]
                additional torsion to be added.

            Returns
            -------
            dict[int, float]
                resulting energies per conformer.
            """

            if torsion_atom_ids is not None:
                self.torsion_atom_ids = torsion_atom_ids

            if _additional_torsions is not None:
                self._additional_torsions = _additional_torsions

            res_rdmols=[]
            for confID in _conf_range:
                off_mol = off_mols[confID]
                
                # Dummy partial charges Todo: rework env.
                off_mol.assign_partial_charges("gasteiger")

                # Build openMM files from off:
                ff = ForceField(self.ff_model)
                off_top = off_mol.to_topology()
                omm_top = off_top.to_openmm()
                conf = off_mol.conformers[0].to_openmm()

                self.omm_top = omm_top
                omm_sys = ff.create_openmm_system(off_top, charge_from_molecules=[off_mol])
                self.omm_sys = omm_sys

                # Build OpenMM System:
                simulation = self.__build_dummy_openmm_simulation(omm_top=omm_top, omm_sys=omm_sys)

                # set the initial positions and velocities
                simulation.context.setPositions(conf)

                rdmol = self.__calculate_openmm_optimized(off_mol=off_mol, simulation=simulation,
                                                          reporter=None,
                                                          nsteps=optimization_nsteps,
                                                          tol_emin=optimization_tol)
                rdmol.SetProp("confID", str(confID))

                if self.torsion_atom_ids is not None:
                    angle = np.round(
                        Chem.rdMolTransforms.GetDihedralDeg(
                            rdmol.GetConformer(), *self.torsion_atom_ids
                        )
                    )
                    rdmol.SetProp("torsion_angle", str(angle))

                res_rdmols.append(rdmol)

            out_df = mols_to_moldf(res_rdmols)

            if off_mol.name is None or off_mol.name == "":
                molIDs  = [f"mol_{ind}" for ind in _conf_range]
            else:
                molIDs = off_mol.name
            out_df["molID"] = molIDs

            if out_file_path is not None:
                store_mol_db(
                    out_df,
                    out_sdf_path=out_file_path
                )

            return out_df

        ## Public
        def calculate_conformer_potentials(
            self,
            mol: Union[pd.DataFrame, Iterable[Chem.Mol], Chem.Mol],
            torsion_atom_ids: list[int] = None,
            out_file_path: str = None,
            _conf_range: list[int] = None,
            _additional_pos_res: list[int] = None,
            _additional_torsions: list[tuple[int, int, int, int, float]] = None,
        ) -> pd.DataFrame:
            """
            Calculate the Torsion profile potentials for a single selected torsion.

            Parameters
            ----------
            mol : Chem.Mol
                molecule of interest
            torsion_atom_ids : list[int]
                torsion indices
            _conf_range : bool, optional
                used to parallelize
            _additional_pos_res: list[int], optional
                additional atoms to be constraint
            Returns
            -------
            pd.DataFrame
                result containing the ANI potentials in kJ/mol
            """

            if (isinstance(mol, Chem.Mol)):
                calculate_mols = [mol]
            elif (isinstance(mol, pd.DataFrame)):
                calculate_mols = mol["ROMol"].to_list()
            elif (isinstance(mol, pd.DataFrame)):
                calculate_mols = mol
            else:
                raise IOError("Need to have an acceptable input type")

            if _conf_range is None:
                _conf_range = list(range(len(calculate_mols)))

            # convert
            off_mols = [Molecule.from_rdkit(m, hydrogens_are_explicit=True) for m in
                       calculate_mols]

            # Calculate / if no optimize, the _optimize_structure attribute will control this
            df = self.__calculate_profile_with_openmm(
                off_mols=off_mols,
                torsion_atom_ids=torsion_atom_ids,
                optimization_nsteps=self._optimize_structure_nsteps,
                optimization_tol=self._optimize_structure_tol,
                out_file_path=out_file_path,
                _conf_range=_conf_range,
                _additional_torsions=_additional_torsions,
            )

            return df

else:
    class OffCalculator(_AbstractCalculator):
        """
        Dummy implementation
        """
        def __init__(
            self,
            ff_model: str = "openff-2.0.0.offxml",
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = 100000,
            optimize_structure_tol: float = 10**-6,
            optimize_structure_write_out: bool = True,
        ) -> None:
            """
            stub as dummy if not in correct env.
            """
            self.ff_model = ff_model
            self._optimize_structure = optimize_structure
            self._optimize_structure_nsteps = optimize_structure_nsteps
            self._optimize_structure_tol = optimize_structure_tol
            # logger.warning("OffCalculator: Could not find openMM")

        def calculate_conformer_potentials(
            self,
            mol: Chem.Mol,
            torsion_atom_ids: list[int] = None,
            out_file_path: str = None,
            _conf_range: list[int] = None,
            _additional_pos_res: list[int] = None,
            _additional_torsions: list[tuple[int, int, int, int, float]] = None,
        ) -> pd.DataFrame:
            """
            stub as dummy if not in correct env.
            """
            raise ImportError("Can not be done, as no openMM in the env!")
