"""
Gaussian Calculator implementation
"""

import os
import tempfile
from typing import Iterable, Union
import tqdm
import pandas as pd

from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolToXYZBlock

try:
    from ..utils.bash import program_binary_found
    import cclib
    if not program_binary_found("g16"):
        raise ImportError("Could not find Gaussian binary")
    found_gaussian_deps = True

except ImportError:
    found_gaussian_deps = False

from ._abstractcalculator import _AbstractCalculator
from ..utils import bash, units
from ..utils import store_mol_db, mols_to_moldf
from ..tools.torsion_profiler.utils import get_angles_of_torsion_profile_mol

if found_gaussian_deps:
    class com_file:
        """
        com file class
        """
        def __init__(
            self,
            basis: str,
            method: str,
            solvation: bool,
            scf: str,
            integration: str,
            num_proc: int,
            mem: str,
            mol: Chem.Mol,
            out_prefix: str,
            out_folder: str,
            torsion_atom_ids: tuple[int, int, int, int] = None,
            additional_torsions: list[tuple[int, int, int, int]] = None,
            nsteps: int = None,
            step_size: float = None,
        ):
            """
            this class can build com files based on input vars.
            Parameters
            ----------
            basis: str
                basis for the calculatoins
            method: str
                functional
            solvation: bool
                shall implicit solvent be added
            scf: str

            integration: str

            num_proc: int
                number of processes
            mem: str
                memory available
            mol: Chem.Mol
                molecule of interest
            out_prefix: str
                output prefix
            out_folder: str
                output folder
            torsion_atom_ids: tuple[int, int, int, int]
                torsion atom ids.
            additional_torsions: list[tuple[int, int, int, int]]
                additional torsion atom ids.
            nsteps: int
                number of steps in the optimization
            step_size: float
                step sizein optimization

            """
            self.basis = basis
            self.method = method
            self.scf = scf
            self.integration = integration
            self.num_proc = num_proc
            self.mem = mem
            self.solvation = solvation
            self.mol = mol
            self.out_prefix = out_prefix
            self.out_folder = out_folder
            self.torsion_atom_ids = torsion_atom_ids
            self.additional_torsions = additional_torsions
            self.nsteps = nsteps
            self.step_size = step_size

        @property
        def optimize_structure(self) -> bool:
            """
            getter for optimiye structure flag
            """
            return self._optimize_structure

        def gen_com_file_sp(self):
            """
            constrcuts the com file for a single point
            """
            with open(os.path.join(self.out_folder, self.out_prefix + ".com"), "w") as f:
                f.write("%nproc=" + str(self.num_proc) + "\n")
                f.write("%mem=" + self.mem + "\n")
                f.write("%chk=" + self.out_prefix + ".chk" + "\n")
                f.write(
                    "#  " + self.method + "/" + self.basis + "\t"
                    "scf=" + self.scf + "\t" + "int=" + self.integration + "\t" + "\n\n"
                )
                f.write(Chem.MolToSmiles(self.mol) + "\n\n")
                f.write(str(Chem.GetFormalCharge(self.mol)) + " 1 \n")
                self.mol.SetProp("_Name", "")
                meta_coordinates = MolToXYZBlock(self.mol)
                to_avoid = str(self.mol.GetNumAtoms()) + "\n\n"
                f.write(meta_coordinates[len(to_avoid) :])
                f.write("\n")

        def gen_com_file_opt(self):
            """
            constructs the com file for an optimization
            """
            with open(os.path.join(self.out_folder, self.out_prefix + ".com"), "w") as f:
                f.write("%nproc=" + str(self.num_proc) + "\n")
                f.write("%mem=" + self.mem + "\n")
                f.write("%chk=" + self.out_prefix + ".chk" + "\n")
                f.write(
                    "#  " + self.method + "/" + self.basis + "\t"
                    "scf=" + self.scf + "\t" + "int=" + self.integration + "\t"
                )
                if self.torsion_atom_ids is None:
                    f.write("Opt=(MaxCycles=500) iop(1/152=500)" + "\n\n")
                else:
                    f.write("Opt=(AddRedundant,MaxCycles=500) iop(1/152=500)" + "\n\n")
                f.write(Chem.MolToSmiles(self.mol) + "\n\n")
                f.write(str(Chem.GetFormalCharge(self.mol)) + " 1 \n")
                self.mol.SetProp("_Name", "")
                meta_coordinates = MolToXYZBlock(self.mol)
                to_avoid = str(self.mol.GetNumAtoms()) + "\n\n"
                f.write(meta_coordinates[len(to_avoid) :])
                f.write("\n")
                if self.torsion_atom_ids is not None:
                    f.write(
                        "D "
                        + str(self.torsion_atom_ids[0] + 1)
                        + " "
                        + str(self.torsion_atom_ids[1] + 1)
                        + " "
                        + str(self.torsion_atom_ids[2] + 1)
                        + " "
                        + str(self.torsion_atom_ids[3] + 1)
                        + " F"
                    )

                if self.additional_torsions is not None:
                    for add_tors in self.additional_torsions:
                        f.write(
                            "D "
                            + str(add_tors[0] + 1)
                            + " "
                            + str(add_tors[1] + 1)
                            + " "
                            + str(add_tors[2] + 1)
                            + " "
                            + str(add_tors[3] + 1)
                            + " F"
                        )


    class GaussianCalculator(_AbstractCalculator):
        """
        Implementation of the Gaussian Calculator
        """

        # Coordinate Optimization:
        _optimize_structure: bool
        _optimize_structure_nsteps: int

        def __init__(
            self,
            basis: str = "cc-pVDZ",
            method: str = "wB97XD",
            solvation: bool = False,
            scf: str = "tight",
            integration: str = "finegrid",
            num_proc: int = 8,
            mem: str = "500MW",
            save_sdf: bool = True,
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = None,
            optimize_structure_step_size: float = None,

        ):
            """Constructor

            Parameters
            ----------
            basis : str, optional
                _description_, by default "cc-pVDZ"
            method : str, optional
                The method (DFT functional, Coupled Cluster, Moeller-Plesset of various degree,etc)
                that is used for the single point calculation (the notation used for Gaussian16).
                Other possib , by default "wB97XD", by default "wB97XD"
            solvation : bool, optional
                If True then calculation will be performed with solvation model, if False then in
                vaccum, by default False
            scf : str, optional
                Settings for convergce of the wavefunction, by default "tight"
            integration : str, optional
                Settings for integration grid, by default "finegrid"
            num_proc: int, optional
                Number of cores used per task, by default 8
            mem : str, optional
                Memory used per task, by default "500MW"
            """
            super().__init__()

            self.basis = basis
            self.method = method
            self.scf = scf
            self.integration = integration
            self.num_proc = num_proc
            self.mem = mem
            self.solvation = solvation
            self.save_sdf = save_sdf
            self._optimize_structure = optimize_structure
            self._optimize_structure_nsteps = optimize_structure_nsteps
            self._optimize_structure_step_size = optimize_structure_step_size

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
        def optimize_structure_step_size(self) -> float:
            """
            getter step size of  optimization
            """
            return self._optimize_structure_step_size

        def calculate_conformer_potentials(
            self,
            mol: Union[pd.DataFrame, Iterable[Chem.Mol], Chem.Mol],
            torsion_atom_ids: tuple[int, int, int, int] = None,
            out_file_path: str = None,
            _conf_range: list = None,
            _additional_torsions=None,
        ):
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

            if out_file_path is None:
                tmp_out = tempfile.mkdtemp(prefix="tmp_Gaus")
                out_file_path = f"{tmp_out}/tmp_tp.sdf"
                print("Tmp-dir: ", out_file_path)

            if self._optimize_structure:
                df = self.calculate_optimization(
                    mols=calculate_mols,
                    torsion_atom_ids=torsion_atom_ids,
                    out_file_path=out_file_path,
                    _conf_range=_conf_range,
                    _additional_torsions=_additional_torsions,
                )
            else:
                df = self.calculate_single_point_energy(
                    mols=calculate_mols,
                    torsion_atom_ids=torsion_atom_ids,
                    out_file_path=out_file_path,
                    _out_file_path=out_file_path,
                    _conf_range=_conf_range
                )

            return df

        def calculate_single_point_energy(
            self, mols: Chem.Mol, _out_file_path: str, _conf_range: list = None,
                torsion_atom_ids=None, out_file_path=None,
        ) -> pd.DataFrame:
            """This function calculates the single point energy of a molecule with one or many
            conformers. For each conformer of a molecule the separate single point gaussian
            calculation will be performed. The corresponding files (.log, .chk, .com wil be
            generated for every conformer calculation) will be generated, executed, and finally
            parsed. The parsed data for conformers will be summarized in the .tsv file.

            Parameters
            ----------
            mols : Chem.Mol
                Molecule to calculate (can be only singlet, can contain multiple conformers)
            _out_file_path : str
                Path to the .tsv file that will be saved after calculation is done. In the same
                folder all generated .log, .chk, .com files will be saved too. This path should
                contain the file name and .tsv extension!
            _conf_range : list, optional
                If you don't want to calculate the single point energies of all conformers present
                in a molecule then specify the conformer's numbers of interest, by default None

            Returns
            -------
            pd.DataFrame
                DataFrame with the absolute and relative values of scf energies in kcal/mol
            """

            if _conf_range is None:
                _conf_range = range(len(mols.GetConformers()))

            out_prefix = bash.path.basename(out_file_path).replace(".sdf", "")
            out_folder = bash.path.dirname(out_file_path) + "/"

            result_mols =  []
            for num in tqdm.tqdm(_conf_range):
                tmp_mol = Chem.Mol(mols[num])
                conf_out_file = out_prefix + f"_{num}_conf"

                # create inputs (.com file)
                file = com_file(
                    basis=self.basis,
                    method=self.method,
                    solvation=self.solvation,
                    scf=self.scf,
                    integration=self.integration,
                    num_proc=self.num_proc,
                    mem=self.mem,
                    mol=tmp_mol,
                    out_prefix=conf_out_file,
                    out_folder=out_folder,
                )
                file.gen_com_file_sp()
                os.chdir(out_folder)
                input_com = conf_out_file + ".com"

                # send the jobs locally
                bash.execute("g16 " + input_com)

                # parse the outpuit (.log)
                filename = out_folder + conf_out_file + ".log"
                parser = cclib.io.ccopen(filename)
                data = parser.parse()

                # extract the scf energies
                scf_energies = data.scfenergies * units.ev_to_kcal_per_mol
                scf_energy = scf_energies[0]

                # extract the resulting geometries
                conf = tmp_mol.GetConformer()
                coordinates = data.atomcoords

                for i in range(tmp_mol.GetNumAtoms()):
                    conf.SetAtomPosition(i, coordinates[0, i, :])

                tmp_mol.SetProp("potential_energy", str(scf_energy))
                tmp_mol.SetProp("confID", str(num))
                result_mols.append(tmp_mol)

            # Post
            out_df = mols_to_moldf(result_mols)

            if torsion_atom_ids is not None:
                angles = get_angles_of_torsion_profile_mol(out_df, torsion_atom_ids=torsion_atom_ids)
                out_df["torsion_angle"] = angles

            if out_file_path is not None:
                store_mol_db(
                    out_df,
                    out_sdf_path=out_file_path,
                )

            return out_df

        def calculate_optimization(
            self,
            mols: Chem.Mol,
            out_file_path: str,
            _conf_range: list = None,
            torsion_atom_ids: tuple[int, int, int, int] = None,
            _additional_torsions=None,
        ) -> pd.DataFrame:
            """This function optimizes a molecule with one or many conformers. For each conformer
            of a molecule the separate optimization will be performed.
             The corresponding files (.log, .chk, .com wil be generated for every conformer
             calculation) will be generated, executed, and finally parsed. The parsed data for
             conformers will be summarized in the .tsv file.

            Parameters
            ----------
            mols : Chem.Mol
                Molecule to calculate (can be only singlet, can contain multiple conformers)
            out_file_path : str
                Path to the .tsv file that will be saved after calculation is done. In the same
                folder all generated .log, .chk, .com files will be saved too. This path should
                contain the file name and .tsv extension!
            _conf_range : list, optional
                If you don't want to calculate the single point energies of all conformers present
                in a molecule then specify the conformer's numbers of interest, by default None
            torsion_atom_ids : tuple[int, int, int, int], optional
                 If you want to perform the restrained optimization specify the atoms to freeze

            Returns
            -------
            pd.DataFrame
               DataFrame with the absolute and relative values of scf energies in kcal/mol
            """

            if _conf_range is None:
                _conf_range = range(len(mols.GetConformers()))

            out_prefix = bash.path.basename(out_file_path).replace(".sdf", "")
            out_folder = bash.path.dirname(out_file_path) + "/"

            result_mols = []
            for num in tqdm.tqdm(_conf_range):
                tmp_mol = Chem.Mol(mols[num])
                conf_out_file = out_prefix + f"_{num}_conf"

                # create inputs (.com file)
                file = com_file(
                    basis=self.basis,
                    method=self.method,
                    solvation=self.solvation,
                    scf=self.scf,
                    integration=self.integration,
                    num_proc=self.num_proc,
                    mem=self.mem,
                    mol=tmp_mol,
                    out_prefix=conf_out_file,
                    out_folder=out_folder,
                    torsion_atom_ids=torsion_atom_ids,
                    additional_torsions=_additional_torsions,
                )
                file.gen_com_file_opt()
                os.chdir(out_folder)
                input_com = conf_out_file + ".com"

                # send the jobs locally
                bash.execute("g16 " + input_com)

                # parse the outpuit (.log)
                filename = out_folder + conf_out_file + ".log"
                parser = cclib.io.ccopen(filename)
                data = parser.parse()

                # extract the scf energies
                scf_energies = data.scfenergies * units.ev_to_kcal_per_mol
                scf_energy = scf_energies[-1]

                # extract the resulting geometries
                conf = tmp_mol.GetConformer()
                coordinates = data.atomcoords

                for i in range(tmp_mol.GetNumAtoms()):
                    conf.SetAtomPosition(i, coordinates[-1, i, :])

                tmp_mol.SetProp("potential_energy", str(scf_energy))
                tmp_mol.SetProp("confID", str(num))
                result_mols.append(tmp_mol)

            # Post
            out_df = mols_to_moldf(result_mols)

            if torsion_atom_ids is not None:
                angles = get_angles_of_torsion_profile_mol(out_df, torsion_atom_ids=torsion_atom_ids)
                out_df["torsion_angle"] = angles

            store_mol_db(
                out_df,
                out_sdf_path=out_file_path,
            )

            return out_df

else:
    class GaussianCalculator(_AbstractCalculator):
        """
        Dummy implementation
        """

        optimize_structure: bool = True,
        optimize_structure_nsteps: int = None,
        optimize_structure_step_size: float = None,

        def __init__(
            self,
            basis: str = "cc-pVDZ",
            method: str = "wB97XD",
            solvation: bool = False,
            scf: str = "tight",
            integration: str = "finegrid",
            num_proc: int = 8,
            mem: str = "500MW",
            save_sdf: bool = True,
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = None,
            optimize_structure_step_size: float = None,
        ):
            """
            stub as dummy if not in correct env.
            """
            self.basis = basis
            self.method = method
            self.scf = scf
            self.integration = integration
            self.num_proc = num_proc
            self.mem = mem
            self.solvation = solvation
            self._optimize_structure = optimize_structure
            self.save_sdf = save_sdf
            self.optimize_structure_nsteps = optimize_structure_nsteps
            self.optimize_structure_step_size = optimize_structure_step_size

            # logger.warning("GaussianCalculator: Could not find cclib")

        def calculate_conformer_potentials(
            self,
            mol: Chem.Mol,
            torsion_atom_ids: list[int] = None,
            out_file_path: str = None,
            _conf_range: list[int] = None,
            _additional_pos_res: list[int] = None,
        ) -> pd.DataFrame:
            """
            stub as dummy if not in correct env.
            """
            raise ImportError("Can not be done, as no Gaussian in the env!")
