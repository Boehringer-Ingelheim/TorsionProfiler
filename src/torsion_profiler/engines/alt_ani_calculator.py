"""
ANI Calculator implementation
"""
import tempfile
from typing import Iterable, Union

import numpy as np
import pandas as pd
from rdkit import Chem

try:
    import torch as pt
    from torchani.models import ANI1ccx, ANI2x, BuiltinModel

    model_dict = {"ani2x": ANI2x,
                  "ani1ccx": ANI1ccx}

    from NNPOps import OptimizedTorchANI  # so far was more stable without

    from openmmtorch import TorchForce

    from openmm import unit as omm_unit
    from openmm import Platform
    from openmm import CustomExternalForce, CustomTorsionForce
    from openmm import LangevinMiddleIntegrator, System
    from openmm.app import Simulation, PDBReporter, Topology

    from openff.toolkit.topology import Molecule
    from openff.toolkit.typing.engines.smirnoff import ForceField
    from openff.units import unit

    found_ani_deps = True

except ImportError as err:
    found_ani_deps = False

from ..utils import bash
from ..utils import store_mol_db, mols_to_moldf
from ._abstractcalculator import _AbstractCalculator

if found_ani_deps:
   # Build ANI Class (from torch openMM)

   class _NNP(pt.nn.Module):
       def __init__(self, atomic_numbers: int, model_ff, use_with_openmm: bool = False):

           super().__init__()

           # Get Name of the model
           self.model_name = model_ff.__name__

           # Store the atomic numbers
           self.atomic_numbers = pt.tensor(atomic_numbers).unsqueeze(0)

           # Create an ANI-2x model
           self.model = model_ff(periodic_table_index=True)

           # optional Accelerate the model for openMM
           if use_with_openmm:
               self.__optimize_for_openmm()

       def __optimize_for_openmm(self):
           if hasattr(self, "model"):
               self.model = OptimizedTorchANI(self.model, self.atomic_numbers)
           else:
               raise ValueError("there was no initial model found!")

       def forward(self, positions):

           # Prepare the positions
           positions = positions.unsqueeze(0).float()  # * 10 # nm --> Å

           # Run ANI-2x
           result = self.model((self.atomic_numbers, positions))

           # Get the potential energy
           energy = result.energies[0] * 2625.5  # Hartree --> kJ/mol

           return energy



   class AltAniCalculator(_AbstractCalculator):
        """
        Implementation of the ANI Calculator
        """

        # Coordinate Optimization:
        _optimize_structure: bool
        _optimize_structure_nsteps: int
        _optimize_structure_tol: float
        _optimize_structure_write_out: bool

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

            self._ff_model = ff_model
            self._anitorch_model = model_dict[ff_model]
            self._optimize_structure = optimize_structure
            self._optimize_structure_nsteps = optimize_structure_nsteps
            self._optimize_structure_tol = optimize_structure_tol
            self._optimize_structure_write_out = optimize_structure_write_out

            self._additional_torsions = None
            self.torsion_atom_ids = None

        # Propterties:
        @property
        def ff_model(self) -> str:
            """
            force field model
            """
            return self._ff_model

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

        ## Privates
        def __build_torchani_nn(self, offMol, anitorch_model, use_with_openmm: bool = False):
            atomic_numbers = [atom.atomic_number for atom in offMol.atoms]
            self._nnp = _NNP(atomic_numbers, model_ff=anitorch_model, use_with_openmm=use_with_openmm)


        def __calculate_single_point(self, mols:Iterable[Molecule],
                                     _conf_range: Iterable[int] = None) -> pd.DataFrame:

            if _conf_range is None:
                _conf_range = range(len(mols))

            res_rdmols = []
            for confID in _conf_range:
                off_mol = mols[confID]
                rdmol = Chem.Mol(mols[confID].to_rdkit())
                npconf = off_mol.conformers[0]
                conf = np.array([[p.to(unit.angstrom).magnitude for p in pos] for pos in npconf])
                pos = pt.tensor(conf)
                calc_v = float(self._nnp(pos))

                rdmol.SetProp("confID", str(confID))
                rdmol.SetProp("potential_energy", str(calc_v))
                if not rdmol.HasProp("_Name") or rdmol.GetProp("_Name") == "":
                    rdmol.SetProp("molID", f"mol_{confID}")
                else:
                    n =rdmol.GetProp("_Name")
                    rdmol.SetProp("molID", f"mol_{n}")

                res_rdmols.append(rdmol)

            out_df = mols_to_moldf(res_rdmols)

            return out_df



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
                system.setParticleMass(i, 0 * omm_unit.amu)

            if hasattr(self, "_additional_pos_res") and self._additional_pos_res is not None:
                for i in self._additional_pos_res:
                    system.setParticleMass(i, 0 * omm_unit.amu)

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

            integrator = LangevinMiddleIntegrator(temperature.magnitude,
                                                  friction_coeff.magnitude,
                                                  time_step.magnitude)

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
            simulation.context.setPositions(conf*10)

            simulation.minimizeEnergy(tolerance=tol_emin_ani, maxIterations=nsteps)

            cstate = simulation.context.getState(
                getPositions=True,
                getEnergy=True,
            )

            calc_v = cstate.getPotentialEnergy().value_in_unit(omm_unit.kilocalorie_per_mole)

            if reporter is not None:
                reporter.report(simulation=simulation, state=cstate)

            res_off_mol = Molecule(off_mol)
            res_off_mol.conformers[0] = cstate.getPositions()
            rdmol = off_mol.to_rdkit()
            rdmol.SetProp("potential_energy", str(calc_v))
            return rdmol

        def __calculate_profile_with_openmm(
            self,
            off_mols: Iterable[Molecule],
            torsion_atom_ids: tuple[int, int, int, int],
            optimization_nsteps: int = 10000,
            optimization_tol: float = 10**-5 * unit.kilojoule_per_mole,
            _conf_range: Iterable[int] = None,
            _additional_torsions: list[tuple[int, int, int, int, float]] = None,
        ) -> pd.DataFrame:
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

            tmp_file = tempfile.NamedTemporaryFile(suffix="_model.pt", dir=bash.getcwd(),
                                                   prefix="pyQMLprop_AniCalc_")
            pt.jit.script(self._nnp).save(tmp_file.name)
            force = TorchForce(tmp_file.name)

            # Build openMM files from off:
            ff = ForceField("openff-1.0.0.offxml")
            off_mol = off_mols[0]
            off_mol.assign_partial_charges("mmff94") # Dummy partial charges
            off_top = off_mol.to_topology()
            omm_top = off_top.to_openmm()
            omm_sys = ff.create_openmm_system(off_top, charge_from_molecules=[off_mol])

            conf = off_mol.conformers[0].to_openmm()

            # check/regulate forces in system we only want ANI:
            ## Remove all forces
            [omm_sys.removeForce(0) for f_id in range(omm_sys.getNumForces())]
            ## Remove all Constraints
            [omm_sys.removeConstraint(0) for f_id in range(omm_sys.getNumConstraints())]

            ## Add Ani
            omm_sys.addForce(force)
            ## set constraints on tors atoms:


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
            self.__build_torchani_nn(offMol=off_mols[0], anitorch_model=self._anitorch_model,
                                     use_with_openmm=self.optimize_structure)
            # Calculate
            if self.optimize_structure:
                out_df = self.__calculate_profile_with_openmm(
                    off_mols=off_mols,
                    torsion_atom_ids=torsion_atom_ids,
                    optimization_nsteps=self._optimize_structure_nsteps,
                    optimization_tol=self._optimize_structure_tol,
                    _conf_range=_conf_range,
                    _additional_torsions=_additional_torsions,
                )
            else:
                out_df = self.__calculate_single_point(mols=off_mols, _conf_range=_conf_range)

            if out_file_path is not None:
                store_mol_db(
                    out_df, out_sdf_path=out_file_path,
                )

            return out_df

else:

    class AltAniCalculator(_AbstractCalculator):
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
            self._ff_model = ff_model
            self._anitorch_model = model_dict[ff_model]
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
