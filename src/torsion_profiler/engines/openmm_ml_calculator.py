"""
ANI Calculator implementation
"""
import shutil
from typing import Iterable, Union

import numpy as np
import pandas as pd
from rdkit import Chem

try:
    from openmmml import MLPotential

    from openmm import unit
    from openmm import CustomExternalForce, CustomTorsionForce
    from openmm import LangevinMiddleIntegrator, System
    from openmm import Platform
    from openmm.app import Simulation, PDBReporter, Topology

    from openff.toolkit.topology import Molecule
    from openff.toolkit.typing.engines.smirnoff import ForceField

    found_ani_deps = True

except ImportError:
    found_ani_deps = False

from ..utils import bash
from ..utils import store_mol_db, mols_to_moldf
from ..data.ml_models import mace_models, mace_model_root

from ._abstractcalculator import _AbstractCalculator

if found_ani_deps:
   # Build ANI Class (from torch openMM)
    class _OpenMMMLBaseCalculator(_AbstractCalculator):
        """
        Implementation of the OpenMM-ML Calculator
        """

        # Coordinate Optimization:
        _optimize_structure: bool
        _optimize_structure_nsteps: int
        _optimize_structure_tol: float
        _optimize_structure_write_out: bool

        # Magics
        def __init__(
            self,
            ff_model: str = "MACE-OFF23_small",
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = 100000,
            optimize_structure_tol: float = 10**-6,
            optimize_structure_write_out: bool = True,
        ):
            """
                Initialize the class

            Parameters
            ----------
            ff_model: BuiltinModel
                the ff model to be used.
            optimize_structure: bool
                optimize the structure of the molecule?
            optimize_structure_nsteps
                number of maximal steps for the optimization.
            optimize_structure_tol
                convergence tolerance
            optimize_structure_write_out: bool
                write the output into a file.
            """

            self.ff_model = ff_model
            self._optimize_structure = optimize_structure
            self._optimize_structure_nsteps = optimize_structure_nsteps
            self._optimize_structure_tol = optimize_structure_tol
            self._optimize_structure_write_out = optimize_structure_write_out

            self._additional_torsions = None
            self.torsion_atom_ids = None

            # Propterties:
        @property
        def optimize_structure(self) -> bool:
            """
            getter shall the structure be optimized
            """
            return self._optimize_structure

        @property
        def optimize_structure_nsteps(self) -> int:
            """
            getter max number of steps
            """
            return self._optimize_structure_nsteps

        @property
        def optimize_structure_tol(self) -> float:
            """
            getter convergence criterium
            """
            return self._optimize_structure_tol

        @property
        def optimize_structure_write_out(self) -> bool:
            """
            getter shall structures be written out?
            """
            return self._optimize_structure_write_out

        # Functions
        def __set_pos_constraints_for_torsion(
            self, torsion_atom_ids: tuple[int, int, int, int], system: System
        ):
            """
                position constraints for the torsion atoms

            Parameters
            ----------
            torsion_atom_ids: tuple[int, int, int, int]
            system: System

            """
            for i in torsion_atom_ids:
                system.setParticleMass(i, 0 * unit.amu)

            if hasattr(self, "_additional_pos_res") and self._additional_pos_res is not None:
                for i in self._additional_pos_res:
                    system.setParticleMass(i, 0 * unit.amu)

        def __set_pos_restraints_for_torsion(
            self, torsion_atom_ids: tuple[int, int, int, int], system: System, conf: unit.Quantity
        ):
            """
                position restraints for the torsion atoms

            Parameters
            ----------
            torsion_atom_ids: tuple[int, int, int, int]
            system: System
            conf: unit.Quantity

            Returns
            -------
            System
                openFF System

            """
            restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
            restraint.addGlobalParameter("k", 10**4 * unit.kilojoules_per_mole / unit.nanometer)
            restraint.addPerParticleParameter("x0")
            restraint.addPerParticleParameter("y0")
            restraint.addPerParticleParameter("z0")

            for atom_idx in torsion_atom_ids:
                restraint.addParticle(atom_idx, conf[atom_idx])

            if hasattr(self, "_additional_pos_res") and self._additional_pos_res is not None:
                for atom_idx in self._additional_pos_res:
                    restraint.addParticle(atom_idx, conf[atom_idx])
            system.addForce(restraint)

            return system

        def __set_tors_restraints_for_torsion(
            self, torsion_atom_ids: tuple[int, int, int, int], angle: float, k: float = 1
        ):
            """
                torsion restraint for the torsion atoms

            Parameters
            ----------
            torsion_atom_ids: tuple[int, int, int, int]
            angle: float
            k: float

            Returns
            -------
            CustomTorsionForce
                the generated force for restraining.
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
                build openn MM simulation system.

            Parameters
            ----------
            omm_top: Topology
            omm_sys: System

            Returns
            -------
            Simulation
                simulation that shall be carried out.
            """
            # Settings
            temperature = 298.15 * unit.kelvin
            friction_coeff = 1 / unit.picosecond
            time_step = 1 * unit.femtosecond

            integrator = LangevinMiddleIntegrator(temperature, friction_coeff, time_step)

            avail_plats = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
            print(avail_plats)
            print(Platform.getPluginLoadFailures())

            try:
                platform = Platform.findPlatform(avail_plats)
                print(f"going with {platform}")
            except Exception as err:
                print("Platform err: ", "\n".join(err.args))
                platform=None
            return Simulation(omm_top, omm_sys, integrator, platform)

        def __calculate_openmm_optimized(
            self,
            off_mol: Molecule,
            simulation: Simulation,
            reporter: PDBReporter = None,
            nsteps: int = 10000,
            tol_emin_ani: float = 10**-5 * unit.kilojoule_per_mole,
            _conf_range: Iterable[int] = None,
        ) -> dict[int, float]:
            """

            Parameters
            ----------
            off_mol: Molecule
            simulation: Simulation
            reporter: PDBReporter
            nsteps: int
            tol_emin_ani: float
            _conf_range: Iterable[int]

            Returns
            -------
            dict[int, float]

            """

            conf = off_mol.conformers[0].to_openmm()


            simulation.context.reinitialize()
            simulation.context.setPositions(conf)

            if self.optimize_structure:
                simulation.minimizeEnergy(tolerance=tol_emin_ani, maxIterations=nsteps)

            cstate = simulation.context.getState(
                getPositions=True,
                getEnergy=True,
            )

            calc_v = cstate.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)

            if reporter is not None:
                reporter.report(simulation=simulation, state=cstate)

            res_off_mol = Molecule(off_mol)
            res_off_mol.conformers[0] = cstate.getPositions()
            rdmol= off_mol.to_rdkit()
            rdmol.SetProp("potential_energy", str(calc_v))
            return rdmol

        def __calculate_profile_with_openmm(
            self,
            off_mols: Molecule,
            torsion_atom_ids: tuple[int, int, int, int],
            out_file_path: str = None,
            optimization_nsteps: int = 10000,
            optimization_tol: float = 10**-5 * unit.kilojoule_per_mole,
            _conf_range: Iterable[int] = None,
            _additional_torsions: list[tuple[int, int, int, int, float]] = None,
        )-> dict[int, float]:
            """
                Build up an openMM system and trigger the calculations.

            Parameters
            ----------
            off_mol: Molecule
            torsion_atom_ids: tuple[int, int, int, int]
            optimization_nsteps: int
            optimization_tol: float
            _conf_range: Iterable[int]
            out_file_path: str
            _additional_torsions: list[tuple[int, int, int, int, float]]

            Returns
            -------
            dict[int, float]
                returns the calculated results
            """
            if torsion_atom_ids is not None:
                self._torsion_angles = []
                self._additional_torsions = _additional_torsions

            # Build openMM files from off:
            off_mol = off_mols[0]
            off_top = off_mol.to_topology()
            omm_top = off_top.to_openmm()
            conf = off_mol.conformers[0].to_openmm()

            #
            print(self.ff_model)
            if(self.ff_model in mace_models):
                from openmmml.models import macepotential
                MLPotential.registerImplFactory("mace",
                                                macepotential.MACEPotentialImplFactory())
                potential = MLPotential("mace", modelPath=mace_models[self.ff_model])
            else:
                potential = MLPotential(self.ff_model)
            omm_sys = potential.createSystem(omm_top)

            # Build OpenMM System:
            simulation = self.__build_dummy_openmm_simulation(omm_top=omm_top, omm_sys=omm_sys)

            ## AddTors res
            if torsion_atom_ids is not None:
                self.__set_pos_constraints_for_torsion(
                    torsion_atom_ids=torsion_atom_ids, system=simulation.system,
                )

            if _additional_torsions is not None:
                for i, additional_torsion in enumerate(_additional_torsions, start=1):
                    self.__set_pos_constraints_for_torsion(
                        torsion_atom_ids=additional_torsion, system=simulation.system,
                    )

            res_rdmols = []
            for confID in _conf_range:
                off_mol = off_mols[confID]

                # set the initial positions and velocities
                simulation.context.setPositions(conf)

                rdmol = self.__calculate_openmm_optimized(
                    off_mol=off_mol,
                    simulation=simulation,
                    tol_emin_ani=optimization_tol,
                    nsteps=optimization_nsteps,
                    _conf_range=_conf_range,
                )
                rdmol.SetProp("confID", str(confID))

                if torsion_atom_ids is not None:
                    angle = np.round(
                        Chem.rdMolTransforms.GetDihedralDeg(
                            rdmol.GetConformer(), * torsion_atom_ids
                        )
                    )
                    rdmol.SetProp("torsion_angle", str(angle))

                res_rdmols.append(rdmol)

            out_df = mols_to_moldf(res_rdmols)

            if off_mol.name is None or off_mol.name == "":
                molIDs = [f"mol_{ind}" for ind in _conf_range]
            else:
                molIDs = off_mol.name
            out_df["molID"] = molIDs

            if out_file_path is not None:
                store_mol_db(
                    out_df, out_sdf_path=out_file_path,
                )

            if bash.path.isfile("animodel.pt"):
                bash.remove("animodel.pt")

            return out_df

        ## Public
        def calculate_conformer_potentials(
            self,
            mol: Union[pd.DataFrame, Iterable[Chem.Mol], Chem.Mol],
            torsion_atom_ids: tuple[int, int, int, int] = None,
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
            # Create an instance of ani model

            # Calculate
            ani_calc_v = self.__calculate_profile_with_openmm(
                off_mols=off_mols,
                torsion_atom_ids=torsion_atom_ids,
                optimization_nsteps=self._optimize_structure_nsteps,
                optimization_tol=self._optimize_structure_tol,
                _conf_range=_conf_range,
                out_file_path=out_file_path,
                _additional_torsions=_additional_torsions,
            )
            return ani_calc_v


    class MACECalculator(_OpenMMMLBaseCalculator):
        """
         Wrapper to build the MACE Calculator
        """

        # Magics
        def __init__(
            self,
            ff_model: str = "MACE-OFF23_small",
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = 100000,
            optimize_structure_tol: float = 10**-6,
            optimize_structure_write_out: bool = True,
        ):
            """
                Initialize the class

            Parameters
            ----------
            ff_model: BuiltinModel
                the ff model to be used.
            optimize_structure: bool
                optimize the structure of the molecule?
            optimize_structure_nsteps
                number of maximal steps for the optimization.
            optimize_structure_tol
                convergence tolerance
            optimize_structure_write_out: bool
                write the output into a file.
            """
            super().__init__(
                ff_model=ff_model,
                optimize_structure=optimize_structure,
                optimize_structure_nsteps=optimize_structure_nsteps,
                optimize_structure_tol=optimize_structure_tol,
                optimize_structure_write_out=optimize_structure_write_out,
            )


    class AniCalculator(_OpenMMMLBaseCalculator):
        """
        Wrapper to build the ANI Calculator
        """
        # Magics
        def __init__(
            self,
            ff_model: str = "ani2x",
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = 100000,
            optimize_structure_tol: float = 10**-6,
            optimize_structure_write_out: bool = True,
        ):
            """
                Initialize the class

            Parameters
            ----------
            ff_model: BuiltinModel
                the ff model to be used.
            optimize_structure: bool
                optimize the structure of the molecule?
            optimize_structure_nsteps
                number of maximal steps for the optimization.
            optimize_structure_tol
                convergence tolerance
            optimize_structure_write_out: bool
                write the output into a file.
            """
            super().__init__(
                ff_model=ff_model,
                optimize_structure=optimize_structure,
                optimize_structure_nsteps=optimize_structure_nsteps,
                optimize_structure_tol=optimize_structure_tol,
                optimize_structure_write_out=optimize_structure_write_out,
            )

else:

    class AniCalculator(_AbstractCalculator):
        """
        Dummy implementation
        """
        def __init__(
            self,
            ff_model: str = "ani2x",
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = 100000,
            optimize_structure_tol: float = 10**-6,
            optimize_structure_write_out: bool = True,
        ):
            """
            stub as dummy if not in correct env.
            """
            self.ff_model = ff_model
            self._optimize_structure = optimize_structure
            self._optimize_structure_nsteps = optimize_structure_nsteps
            self._optimize_structure_tol = optimize_structure_tol
            self._optimize_structure_write_out = optimize_structure_write_out

            # logger.warning("AniCalculator: Could not find ani-torch")

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
            raise ImportError("Can not be done, as no ANI in the env!")

    class MACECalculator(_AbstractCalculator):
        """
        Dummy implementation
        """
        def __init__(
            self,
            ff_model: str = "MACE-OFF23_small",
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = 100000,
            optimize_structure_tol: float = 10**-6,
            optimize_structure_write_out: bool = True,
        ):
            """
            stub as dummy if not in correct env.
            """
            self.ff_model = ff_model
            self._optimize_structure = optimize_structure
            self._optimize_structure_nsteps = optimize_structure_nsteps
            self._optimize_structure_tol = optimize_structure_tol
            self._optimize_structure_write_out = optimize_structure_write_out

            # logger.warning("AniCalculator: Could not find ani-torch")

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
            raise ImportError("Can not be done, as no ANI in the env!")