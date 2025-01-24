"""    import xtb

XTB Calculator implementation
"""

import os
import tempfile
from typing import Union, Iterable

import tqdm
import pandas as pd
from rdkit import Chem

try:
    from ..utils.bash import program_binary_found

    if not program_binary_found("xtb"):
        raise ImportError("Could not find XTB binary")

    found_xtb_deps = True

except ImportError as err:
    found_xtb_deps = False

from ..utils import bash
from ..utils.units import hatree_to_kcal_per_mol
from ..utils import store_mol_db, mols_to_moldf

from ._abstractcalculator import _AbstractCalculator

if found_xtb_deps:

    class XtbCalculator(_AbstractCalculator):
        """
        Implementation of the XTB Calculator
        """
        def __init__(
            self,
            method: int = 2,
            solvation: bool = False,
            acc: float = "1",
            iterations: int = 150,
            opt_level: str = "tight",
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = None,
            optimize_structure_step_size: float = None,
            optimize_structure_write_out: bool = True,
            _omp_mem: str = "4G",
            num_proc: int = 1,
        ):
            """constructor

            Parameters
            ----------
            method : int, optional
                Choose the method according to the xTB documentation (0,1,2...) , by default
                2 (2==GFN2-xTB)
            solvation : bool, optional
                If True then calculation will be performed with solvation model, if False then in
                vaccum, by default False
            acc : float, optional
                The accuracy determines the integral screening thresholds and the SCC convergence
                criteria and can be adjusted continuous in a range from 0.0001 to 1000, where
                tighter criteria are set for lower values of accuracy., by default "1"
            num_proc : int, optional
                Number of cores used, by default 8
            iterations : int, optionalR
                The number of iterations allowed for the SCC calculation, by default 250
            opt_level : str, optional
                Optimization level as specified in xTB documentation
                (https://xtb-docs.readthedocs.io/en/latest/optimization.html#id5),
                by default "tight"
            optimize_structure_write_out : bool, optional
                If you wamt to save the optimized/calculated geometries as sdf. file set True,
                by default True
            optimize_structure_nsteps : int, optional
                If you want to perform the sequential scan specify the number of steps along the
                scanned coordinate, by default None
            optimize_structure_step_size : float, optional
                If you want to perform the sequential scan specify the step size along the scanned
                coordinate, by default None
            optimize_structure : bool, optional
                If you want to optimize the structure set True, by default False
            """
            super().__init__()
            self.method = method
            self.acc = acc
            self.num_proc = num_proc
            self.iterations = iterations
            self.solvation = solvation
            self.opt_level = opt_level
            self._optimize_structure = optimize_structure
            self._optimize_structure_write_out = optimize_structure_write_out
            self._optimize_structure_nsteps = optimize_structure_nsteps
            self.optimize_structure_step_size = optimize_structure_step_size
            self._omp_mem = _omp_mem


        def calculate_conformer_potentials(
            self,
            mol: Union[pd.DataFrame, Iterable[Chem.Mol], Chem.Mol],
            torsion_atom_ids: tuple[int, int, int, int] = None,
            out_file_path: str = None,
            _conf_range: list[int] = None,
            _additional_torsions: list[tuple[int, int, int, int, float]] = None,
        ) -> pd.DataFrame:
            """
            evaluate the conformer potentials.

            Parameters
            ----------
            mol : Union[pd.DataFrame, Iterable[Chem.Mol], Chem.Mol]
                Molecule to calculate (can be only singlet, can contain multiple conformers)
            torsion_atom_ids: tuple[int, int, int, int]
                atom ids definein gthe torsion in the molecule.
            out_file_path : str
                Path to the .tsv file that will be saved after calculation is done. In the same
                folder all generated .log, .chk, .com files will be saved too. This path should
                contain the file name and .tsv extension!
            _conf_range : list, optional
                If you don't want to calculate the single point energies of all conformers present
                in a molecule then specify the conformer's numbers of interest, by default None
            _additional_torsions: list[tuple[int, int, int, int, float]]
                additional restraints


            Returns
            -------
            pd.DataFrame
                DataFrame with the absolute and relative values of scf energies in kcal/mol
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
                    mols=calculate_mols, out_file_path=out_file_path, _conf_range=_conf_range
                )

            return df

        def calculate_single_point_energy(
            self,
            mols: Chem.Mol,
            out_file_path: str = None,
            _conf_range: list = None,
            clean: bool = True,
            verbose: bool = False,
        ) -> pd.DataFrame:
            """This function calculates the single point energy of a molecule with one or many
            conformers. For each conformer of a molecule the separate single point xTB
            calculation will be performed. The parsed data for conformers will be summarized in
            the .tsv file.

            Parameters
            ----------
            mol : Chem.Mol
                Molecule to calculate (can be only singlet, can contain multiple conformers)
            out_file_path : str
                Path to the .tsv file that will be saved after calculation is done. This path
                should contain the file name and .tsv extension!
            _conf_range : list, optional
                If you don't want to calculate the single point energies of all conformers present
                in a molecule then specify the conformer's numbers of interest, by default None
            clean : bool, optional
                If you want to clean the temporarily generated .sdf files (input and output sdf for
                every single conformer) set True, by default True
            verbose : bool, optional
                by default, False

            Returns
            -------
            pd.DataFrame
                DataFrame with the absolute and relative values of scf energies in kcal/mol
            """

            del_temp = False
            if out_file_path is None:
                del_temp = True
                out_dir = tempfile.mkdtemp(prefix="tmp_xtb")
                out_file_path = out_dir + "/tmp.sdf"
                if os.path.exists(out_file_path):
                    raise ValueError("temporary file exists already! will not overwrite")
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)


            # Prepare the molecules and folders for calculation
            out_prefix = bash.path.basename(out_file_path).replace(".sdf", "")
            out_folder = bash.path.dirname(out_file_path) + "/"
            orig_dir = bash.getcwd()
            bash.chdir(out_folder)

            res_rdmols = []
            for num in tqdm.tqdm(_conf_range):
                if verbose:
                    print("I am calculating conformer ", num)

                # Prepare the molecules and folders for calculation
                mol = mols[num]
                out_mol = Chem.Mol(mol)  # copy_of_mol
                mol_charge = Chem.GetFormalCharge(mol)

                # create separate input .sdf file for every single conformer
                input_file = "tmp_" + out_prefix + f"_{num}_conf_input_geom.xyz"
                out_file = "tmp_" + out_prefix + f"_{num}_conf.log"

                f = open(os.path.join(out_folder, input_file), "w")
                f.writelines(Chem.MolToXYZBlock(mol))
                f.close()

                # send the job locally
                conda_env = os.environ["CONDA_PREFIX"]
                print(os.getcwd())
                try:
                    bash.execute(
                        f"{conda_env}/bin/xtb {input_file} --chrg {mol_charge} --uhf 0 "
                        f"--acc {self.acc} "
                        f"--iterations {self.iterations} --gfn {self.method} "
                        f"--parallel {self.num_proc} > {out_file}"
                    )
                except Exception as err:
                    bash.chdir(orig_dir)
                    raise err
                # parse log file and convert scf energy from Hartree to kcal/mol and save it
                scf_energy = self.extract_final_scf_energy(f"{out_folder}/{out_file}")

                out_mol.SetProp("potential_energy", str(scf_energy))
                out_mol.SetProp("confID", str(num))

                res_rdmols.append(out_mol)

                if verbose:
                    print("Ended the conformer ", num)
                if clean:
                    bash.remove(input_file)

            # Post
            out_df = mols_to_moldf(res_rdmols)

            ## save sdf:
            if self.optimize_structure_write_out:
                store_mol_db(
                    df_mols=out_df, out_sdf_path=os.path.join(out_folder, out_prefix + "_tp.sdf")
                )

            if del_temp:
                bash.rmtree(out_dir)

            bash.chdir(orig_dir)
            return out_df

        def calculate_optimization(
            self,
            mols: Chem.Mol,
            torsion_atom_ids: tuple[int, int, int, int] = None,
            out_file_path: str = None,
            _conf_range: list = None,
            clean: bool = True,
            verbose: bool = False,
            _additional_torsions=None,
        ) -> pd.DataFrame:
            """This function optimizes a molecule with one or many conformers. For each conformer
            of a molecule the separate optimization will be performed.
            The corresponding files (.log, .sdf, .inp will be generated, executed, and finally
            parsed. The parsed data for conformers will be summarized in the .tsv file.

            Parameters
            ----------
            mol : Chem.Mol
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
            clean : bool, optional
                If you want to clean the temporarily generated .sdf files (input and output sdf for
                every single conformer) set True, by default True
            verbose : boolean, optional
                by default, False

            Returns
            -------
            pd.DataFrame
            DataFrame with the absolute and relative values of scf energies in kcal/mol
            """
            del_temp = False
            if out_file_path is None:
                del_temp = True
                out_dir = tempfile.mkdtemp(prefix="tmp_xtb")
                out_file_path = f"{out_dir}/tmp.sdf"
                if os.path.exists(out_file_path):
                    raise ValueError("temporary file exists already! will not overwrite")
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)

            # Prepare the molecules and folders for calculation
            out_prefix = bash.path.basename(out_file_path).replace(".sdf", "")
            out_folder = bash.path.dirname(out_file_path) + "/"
            orig_dir = bash.getcwd()
            bash.chdir(out_folder)

            if torsion_atom_ids is not None:
                fix_atoms = list(map(lambda x: str(x + 1), torsion_atom_ids))
                if _additional_torsions is not None:
                    for at in _additional_torsions:
                        t = at[:4]
                        fix_atoms.extend(list(map(lambda x: str(x + 1), t)))
                fix_atoms = list(sorted(list(set(fix_atoms))))
                atoms_to_fix = ",".join(fix_atoms)

                with open(os.path.join(out_folder, "xtb.inp"), "w") as f:
                    f.write("$fix \n")
                    f.write(" atoms: " + atoms_to_fix + "\n")
                    f.write("$end")

            res_rdmols = []
            for num in tqdm.tqdm(_conf_range):
                if verbose:
                    print("I am calculating conformer ", num)

                # Prepare the molecules and folders for calculation
                mol = mols[num]
                out_mol = Chem.Mol(mol)  # copy_of_mol
                mol_charge = Chem.GetFormalCharge(mol)

                # create separate input .sdf file for every single conformer
                input_file = "tmp_" + out_prefix + f"_{num}_conf_input_geom.xyz"
                out_file = "tmp_" + out_prefix + f"_{num}_conf.log"
                Chem.MolToXYZFile(mol, input_file)

                # send the job
                conda_env = os.environ["CONDA_PREFIX"]
                prefix = ""

                _out_namespace = "./tmp_" + out_prefix + "_ns"

                # Options
                options = {
                    "chrg": mol_charge,
                    "uhf": 0,
                    "acc": self.acc,
                    "iterations": self.iterations,
                    "gfn": self.method,
                    "namespace": _out_namespace,
                    "opt": self.opt_level,
                }

                if torsion_atom_ids is not None:
                    options["input"] = "xtb.inp"

                if self.num_proc > 1:
                    options["parallel"] = self.num_proc

                # RUN XTB
                options_str = " ".join(["--" + str(k) + " " + str(v) for k, v in options.items()])
                cmd = f"{prefix}{conda_env}/bin/xtb {input_file} {options_str} > {out_file}"
                try:
                    bash.execute(cmd)
                except Exception as err:
                    bash.chdir(orig_dir)
                    raise err

                # extract the computed energy
                tmp_out_coords = os.path.join(out_folder, f"{_out_namespace}.xtbopt.xyz")
                meta_mol = Chem.MolFromXYZFile(tmp_out_coords)
                meta_conf = meta_mol.GetConformer()

                # parse log file and convert scf energy from Hartree to kcal/mol and save it
                scf_energy = self.extract_final_scf_energy(f"{out_folder}/{out_file}")

                # new out_mol
                out_mol.RemoveAllConformers()
                out_mol.AddConformer(meta_conf)
                out_mol.SetProp("potential_energy", str(scf_energy))
                out_mol.SetProp("confID", str(num))
                res_rdmols.append(out_mol)

                if clean:
                    bash.remove(input_file)
                    bash.remove(tmp_out_coords)

            # Post
            out_df = mols_to_moldf(res_rdmols)

            ## save sdf:
            if self.optimize_structure_write_out:
                store_mol_db(
                    df_mols=out_df, out_sdf_path=os.path.join(out_folder, out_prefix + "_tp.sdf"),
                )

            if del_temp:
                bash.rmtree(out_dir)
            bash.chdir(orig_dir)

            return out_df

        def extract_final_scf_energy(self, out_file_path:str)->float:
            to_grep = "TOTAL ENERGY"
            to_parse = ""
            file_read = open(out_file_path, "r")
            lines = file_read.readlines()
            for line in lines:
                if to_grep in line:
                    to_parse = line
            file_read.close()
            scf_energy = float(
                to_parse.replace("TOTAL ENERGY", "")
                .replace("Eh", "")
                .replace(" ", "")
                .replace("|", "")
            )
            scf_energy = scf_energy * hatree_to_kcal_per_mol
            return scf_energy


else:

    class XtbCalculator(_AbstractCalculator):
        """
        Dummy implementation
        """
        def __init__(
            self,
            method: int = 2,
            solvation: bool = False,
            acc: float = "1",
            num_proc: int = 1,
            iterations: int = 250,
            opt_level: str = "tight",
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = None,
            optimize_structure_step_size: float = None,
            optimize_structure_write_out: bool = True,
            _omp_mem: str = "4G",
        ):
            """
            stub as dummy if not in correct env.
            """
            self.method = method
            self.acc = acc
            self.num_proc = num_proc
            self.iterations = iterations
            self.solvation = solvation
            self.opt_level = opt_level
            self._optimize_structure = optimize_structure
            self._optimize_structure_write_out = optimize_structure_write_out
            self._optimize_structure_nsteps = optimize_structure_nsteps
            self.optimize_structure_step_size = optimize_structure_step_size
            self._omp_mem = _omp_mem

        def calculate_conformer_potentials(
            self,
            mol: Union[pd.DataFrame, Iterable[Chem.Mol], Chem.Mol],
            torsion_atom_ids: tuple[int, int, int, int] = None,
            out_file_path: str = None,
            _conf_range: list[int] = None,
            _additional_torsions: list[tuple[int, int, int, int, float]] = None,
        ) -> pd.DataFrame:
            """
            stub as dummy if not in correct env.
            """
            raise ImportError("Can not be done, as no XTB in the env!")
