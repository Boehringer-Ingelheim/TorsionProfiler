"""
Psi4 Calculator implementation
"""

import os
from typing import Union, Iterable

import tqdm
import pandas as pd
from rdkit import Chem

try:
    import psi4
    from cclib.parser import Psi4

    found_psi4_deps = True
except ImportError as err:

    found_psi4_deps = False

from ._abstractcalculator import _AbstractCalculator
from ..utils import bash, units
from ..utils import store_mol_db, mols_to_moldf

if found_psi4_deps:

    class Psi4Calculator(_AbstractCalculator):
        """
        Implementation of the Psi4 Calculator
        """
        def __init__(
            self,
            basis: str = "cc-pVDZ",
            method: str = "wb97x-d",
            solvation: bool = False,
            scf: float = 1e-5,
            geometric: bool = False,
            cartesian_coords: bool = False,
            optimize_structure: bool = True,
            optimize_structure_nsteps: int = None,
            optimize_structure_step_size: float = None,
            optimize_structure_write_out: bool = True,
            num_proc: int = 8,
            mem: str = "500MB",
        ):
            """Constructor

            Parameters
            ----------
            basis : str, optional
                Basis set, by default "cc-pVDZ"
            method : str, optional
                The method (DFT functional, Coupled Cluster, Moeller-Plesset of various degree,etc)
                that is used for the single point/optimization calculation (the notation used
                for Psi4), by default "wb97x-d"
            solvation : bool, optional
                If True then calculation will be performed with solvation model, if False then in
                vaccum, by default False
            scf : float, optional
                Settings for convergce of the wavefunction, by default 1e-6 Hartree
            num_proc : int, optional
                Number of threads, by default 8
            mem : str, optional
                Memory used per task, by default "500 MB"
            optimize_structure_write_out : bool, optional
                If you wamt to save the optimized/calculated geometries as sdf. file set True, by
                default True
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

            self.basis = basis
            self.method = method
            self.scf = scf
            self.num_proc = num_proc
            self.mem = mem
            self.solvation = solvation
            self.geometric = geometric
            self.cartesian_coords = cartesian_coords

            self._optimize_structure = optimize_structure
            self._optimize_structure_write_out = optimize_structure_write_out
            self._optimize_structure_nsteps = optimize_structure_nsteps
            self._optimize_structure_step_size = optimize_structure_step_size


        def calculate_conformer_potentials(
            self,
            mol: Union[pd.DataFrame, Iterable[Chem.Mol], Chem.Mol],
            torsion_atom_ids: tuple[int, int, int, int] = None,
            out_file_path: str = None,
            _conf_range: list[int] = None,
            _additional_torsions: list[tuple[int, int, int, int, float]]=None,
        )->pd.DataFrame:
            """
            helper function for starting the calculator potential energy calculations.

            Parameters
            ----------
            mol: Union[pd.DataFrame, Iterable[Chem.Mol], Chem.Mol]
                molecule of interest
            torsion_atom_ids: tuple[int, int, int, int]
                atom ids defining the torsion
            out_file_path: str
                path to the outfile
            _conf_range: list[int]
                range of conformers to be calculated
            _additional_torsions: list[tuple[int, int, int, int, float]]
                additional restraints?

            Returns
            -------
            pd.Dataframe
                resulting dataframe.
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

            orig_dir = bash.getcwd()
            if out_file_path is None:
                _out_dir = f"{bash.getcwd()}/tmp_psi4"
                out_file_path = f"{_out_dir}/psi4_out.sdf"
                if os.path.exists(out_file_path):
                    raise ValueError("temporary file exists already! will not overwrite")
                if not os.path.exists(_out_dir):
                    os.mkdir(_out_dir)
                os.chdir(_out_dir)

            # solving tmp problems
            out_dir = bash.path.dirname(out_file_path)
            bash.environ["PSI_SCRATCH"] = out_dir
            psi4.core.IOManager.shared_object().set_default_path(out_dir)

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
            bash.chdir(orig_dir)
            return df

        def calculate_single_point_energy(
            self,
            mols: Chem.Mol,
            out_file_path: str,
            _conf_range: list = None,
            verbose: bool = False,
        ) -> pd.DataFrame:
            """This function calculates the single point energy of a molecule with one or many
            conformers. For each conformer of a molecule the separate single point Psi4
            calculation will be performed. The parsed data for conformers will be summarized in the
            .tsv file.

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
            verbose : boolean, optional
                by default, False

            Returns
            -------
            pd.DataFrame
                DataFrame with the absolute and relative values of scf energies in kcal/mol
            """

            # Psi4 general QM settings
            psi4.set_memory(self.mem)
            psi4.set_num_threads(self.num_proc)

            options = {"reference": "rhf", "E_CONVERGENCE": self.scf}

            if self.cartesian_coords:
                options.update({"optking__opt_coordinates": "CARTESIAN"})
            else:
                options.update({"optking__opt_coordinates": "BOTH"})
            psi4.set_options(options)

            if not verbose:
                psi4.core.be_quiet()

            # Prepare the molecules and folders for calculation
            out_prefix = bash.path.basename(out_file_path).replace(".tsv", "")
            out_folder = bash.path.dirname(out_file_path) + "/"

            res_rdmols = []
            for num in tqdm.tqdm(_conf_range):
                if verbose:
                    print("Entered the cycle number ", num)

                out_mol = Chem.Mol(mols[num])  # copy_of_mol
                mol_coords = Chem.MolToXYZBlock(out_mol, confId=num)
                psi4_molecule = psi4.geometry(mol_coords)
                psi4_molecule.set_molecular_charge(Chem.GetFormalCharge(out_mol))

                # send the job locally
                scf_energy = psi4.energy(self.method + "/" + self.basis, molecule=psi4_molecule)

                # convert scf energy from Hartree to kcal/mol and save it
                scf_energy = scf_energy * units.hatree_to_kcal_per_mol
                bash.remove("psi.*")
                bash.remove(f"{out_folder}/psi.*")

                if verbose:
                    print("Ended the cycle number ", num)

                # extract the computed geometries and update the out_mol conformer with
                # new coordinates
                out_mol.SetProp("potential_energy", scf_energy)
                out_mol.SetProp("confID", num)
                res_rdmols.append(out_mol)

            # Post
            ## create DataFrame with computed energies
            out_df = mols_to_moldf(res_rdmols)

            ## save sdf:
            if self.optimize_structure_write_out:
                store_mol_db(
                    df_mols=out_df, out_sdf_path=os.path.join(out_folder, f"{out_prefix}_tp.sdf")
                )

            return out_df


        def calculate_optimization(
            self,
            mols: Chem.Mol,
            out_file_path: str,
            torsion_atom_ids: tuple[int, int, int, int] = None,
            _conf_range: list = None,
            _additional_torsions=None,
            verbose: bool = False,
        ) -> pd.DataFrame:
            """This function optimizes a molecule with one or many conformers. For each conformer
            of a molecule the separate optimization will be performed.
             The corresponding files (.log, .chk, .com wil be generated for every conformer
             calculation) will be generated, executed, and finally parsed. The parsed data for
             conformers will be summarized in the .tsv file.

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
            verbose : boolean, optional
                by default, False

            Returns
            -------
            pd.DataFrame
               DataFrame with the absolute and relative values of scf energies in kcal/mol
            """


            # Psi4 general QM settings
            psi4.set_memory(self.mem)
            psi4.set_num_threads(self.num_proc)

            options = {"reference": "rhf", "E_CONVERGENCE": self.scf}

            if self.cartesian_coords:
                options.update({"optking__opt_coordinates": "CARTESIAN"})
            else:
                options.update({"optking__opt_coordinates": "BOTH"})

            psi4.set_options(options)

            if not verbose:
                psi4.core.be_quiet()

            # Prepare the molecules and folders for calculation
            out_prefix = bash.path.basename(out_file_path).replace(".sdf", "")
            out_folder = bash.path.dirname(out_file_path) + "/"
            res_rdmols = []
            for num in tqdm.tqdm(_conf_range):
                if verbose:
                    print("Entered the cycle number ", num)

                out_mol = Chem.Mol(mols[num])  # copy_of_mol
                out_file = out_file_path.replace(".sdf", "_" + str(num) + ".log")
                psi4.set_output_file(out_file, loglevel=30)

                mol_coords = Chem.MolToXYZBlock(out_mol, confId=num)
                psi4_molecule = psi4.geometry(mol_coords)
                psi4_molecule.set_molecular_charge(Chem.GetFormalCharge(out_mol))
                if verbose:
                    print("I am working in:     ", psi4.core.Molecule.units(psi4_molecule))
                psi4_molecule.print_out_in_angstrom()

                # send the job
                out_dict=[]
                if (torsion_atom_ids is None) and not self.geometric:
                    psi4.set_options(
                        {"optking__opt_coordinates": "BOTH", "optking__geom_maxiter": 50}
                    )
                    out_dict = psi4.optimize(
                        self.method + "/" + self.basis,
                        molecule=psi4_molecule,
                        return_history=True,
                        engine="optking",
                    )

                elif (torsion_atom_ids is None) and self.geometric:
                    # keywords = {"conv" : 1e-6}
                    in_dict = {"molecule": psi4_molecule}
                    out_dict = psi4.optimize(
                        self.method + "/" + self.basis,
                        return_history=True,
                        engine="geometric",
                        optimizer_keywords=in_dict,
                    )

                elif (torsion_atom_ids is not None) and not self.geometric:
                    atom_numbers = [str(x + 1) for x in torsion_atom_ids]
                    if _additional_torsions is not None:
                        for at in _additional_torsions:
                            atom_numbers.extend([str(x + 1) for x in at[:4]])

                    to_freeze = " ".join(atom_numbers)

                    try:
                        psi4.set_options(
                            {
                                "optking__frozen_dihedral": to_freeze,
                                "optking__opt_coordinates": "BOTH",
                                "optking__geom_maxiter": 80,
                            }
                        )
                        out_dict = psi4.optimize(
                            self.method + "/" + self.basis,
                            molecule=psi4_molecule,
                            return_history=True,
                            engine="optking",
                        )
                    except:
                        file = Psi4(out_file)
                        if file is not None:
                            file = Psi4(out_file)
                            data = file.parse()

                            conf = out_mol.GetConformer(num)
                            min_energy_num = data.scfenergies.argmin()
                            coordinates_to_restart = data.atomcoords[min_energy_num, :, :]

                            for i in range(out_mol.GetNumAtoms()):
                                conf.SetAtomPosition(i, coordinates_to_restart[i, :])

                            new_mol_coords = Chem.MolToXYZBlock(out_mol, confId=num)
                            new_psi4_molecule = psi4.geometry(new_mol_coords)
                            new_psi4_molecule.print_out_in_angstrom()

                            new_out_file = out_file.replace(".log", "_restarted.log")
                            psi4.set_output_file(new_out_file, loglevel=30)

                            psi4.set_options(
                                {
                                    "optking__frozen_dihedral": to_freeze,
                                    "optking__opt_coordinates": "BOTH",
                                    "optking__geom_maxiter": 60,
                                    "optking__INTRAFRAG_STEP_LIMIT": 0.05,
                                    "optking__INTERFRAG_STEP_LIMIT": 0.05,
                                }
                            )
                            out_dict = psi4.optimize(
                                self.method + "/" + self.basis,
                                molecule=new_psi4_molecule,
                                return_history=True,
                                engine="optking",
                            )
                elif torsion_atom_ids is not None and self.geometric:
                    in_dict = {
                        "molecule": psi4_molecule,
                        "constraints": {
                            "freeze": [{"type": "dihedral", "indices": torsion_atom_ids}]
                        },
                    }
                    out_dict = psi4.optimize(
                        "scf/cc-pvdz",
                        return_history=True,
                        engine="geometric",
                        optimizer_keywords=in_dict,
                    )

                # extract the computed energies
                # convert scf energy from Hartree to kcal/mol and save it
                scf_energy = out_dict[0] * units.hatree_to_kcal_per_mol

                # extract the computed geometries and update the out_mol conformer with
                # new coordinates
                conf = out_mol.GetConformer()
                coordinates = out_dict[1]["coordinates"][-1].to_array() * units.bohr_to_angstrom
                for i in range(out_mol.GetNumAtoms()):
                    conf.SetAtomPosition(i, coordinates[i, :])
                out_mol.SetProp("potential_energy", scf_energy)
                out_mol.SetProp("confID", num)
                res_rdmols.append(out_mol)

            # Post
            ## create DataFrame with computed energies
            out_df = mols_to_moldf(res_rdmols)

            ## save sdf:
            if self.optimize_structure_write_out:
                store_mol_db(
                    df_mols=out_df,
                    out_sdf_path=os.path.join(out_folder, f"{out_prefix}_tp.sdf")
                )

            return out_df


else:

    class Psi4Calculator(_AbstractCalculator):
        """
        Dummy implementation
        """
        def __init__(
            self,
            basis: str = "cc-pVDZ",
            method: str = "wb97x-d",
            solvation: bool = False,
            scf: float = 1e-5,
            num_threads: int = 8,
            mem: str = "500MB",
            save_sdf: bool = True,
            nsteps: int = None,
            step_size: float = None,
            _optimize_structure: bool = True,
            geometric: bool = False,
            cartesian_coords: bool = False,
        ):
            """
            stub as dummy if not in correct env.
            """
            self.basis = basis
            self.method = method
            self.scf = scf
            self.num_threads = num_threads
            self.mem = mem
            self.solvation = solvation
            self._optimize_structure = _optimize_structure
            self.save_sdf = save_sdf
            self.nsteps = nsteps
            self.step_size = step_size
            self.geometric = geometric
            self.cartesian_coords = cartesian_coords
            # logger.warning("Psi4Calculator: Could not find psi4")

        def calculate_conformer_potentials(
            self,
            mol: Chem.Mol,
            torsion_atom_ids: list[int] = None,
            _conf_range: list[int] = None,
            _out_file_path: str = None,
            _additional_pos_res: list[int] = None,
            _additional_torsions: list[tuple[int, int, int, int, float]] = None,
        ) -> pd.DataFrame:
            """
            stub as dummy if not in correct env.
            """
            raise ImportError("Can not be done, as no psi4 in the env!")
