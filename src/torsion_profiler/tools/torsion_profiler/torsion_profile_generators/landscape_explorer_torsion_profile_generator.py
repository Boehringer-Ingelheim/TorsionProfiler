"""
Landscape explorer generating input profile
"""
import logging
from copy import deepcopy
from typing import Iterable
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import argrelextrema

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetDihedralDeg

from ....utils import mols_to_moldf

from ._abstract_torsion_profile_generator import _AbstractTorsionProfileGenerator

log = logging.getLogger(__name__)


def thread_landscape_exploration_2d(
        job_id: int,
        m: Chem.Mol,
        t1: tuple[int, int, int, int],
        _additional_pos_res: list[int] = None,
        _additional_torsions: list[tuple[int, int, int, int, float]] = None,
        _align_mols: bool = True,
        tpg_ks: dict[str, any] = None,
) -> tuple[Chem.Mol, list[float], dict[str, any]]:
    """
    thread helper to parallelize 2D landscaping
    Parameters
    ----------
    job_id:int
        number of this job
    m: Chem.Mol
        target molecule
    t1: tuple[int, int, int, int]
        atom ids, defining a torsion
    _additional_pos_res:list[int]
        additional position restraints
    _additional_torsions: list[tuple[int, int, int, int, float]]
        add additional torsions to the optimization
    _align_mols: bool
        shall the mols be aligned?
    tpg_ks: dict[str, any]
        vars of the parent class

    Returns
    -------
    tuple[Chem.Mol, list[float], dict[str, any]]
        pt_tp, merged_energies, vars(tpg)

    """
    tpg = LandscaperTorsionProfileGenerator()
    for k, v in tpg_ks.items():
        setattr(tpg, k, v)

    angle_range = np.round(np.linspace(-180, 180, tpg.n_measurements))  # not  a setting currently
    tpg.iteration_min = []
    tpg.iteration_max = []
    tpg.iteration_v = []
    tpg.iteration_profile_mol = []
    tpg.orig_torsion_atom_ids = deepcopy(t1)

    # Inital run
    opt_tp = tpg._iteration_generate_profile(
        m,
        t1,
        angle_range=angle_range,
        _additional_pos_res=_additional_pos_res,
        _additional_torsions=_additional_torsions,
    )
    merged_energies = tpg.ff_energies

    tpg.iteration_v.append(merged_energies)
    tpg.iteration_profile_mol.append(opt_tp)

    # Explore potential landscape
    for it in range(tpg.exploration_iterations):
        c, new_merged_energies = tpg._optimize_landscape_from_minima(
            opt_tp,
            t1,
            merged_energies,
            _additional_torsions=_additional_torsions,
        )
        tpg.iteration_v.append(new_merged_energies)
        tpg.iteration_profile_mol.append(opt_tp)

        # Convergence Crit
        if (
                tpg.exploration_min_iterations <= it
                and tpg.exploration_convergence
                and np.allclose(new_merged_energies, merged_energies,
                                rtol=tpg.exploration_convergence_tol)
        ):
            merged_energies = np.array(new_merged_energies)

            log.info("converged in: ", it + 1)
            break
        merged_energies = np.array(new_merged_energies)
        t1 = t1[::-1]  # switch rotation dir for ehanced sampling

    # Final curve extrema
    minima, maxima = tpg._find_curve_extrema(merged_energies)
    tpg.iteration_min.append(minima)
    tpg.iteration_max.append(maxima)

    if _align_mols:
        torsion_atom_id_map = list(zip(t1[:3], t1[:3]))
        align_f = lambda m2: Chem.rdMolAlign.AlignMol(prbMol=m2, refMol=m,
                                                     atomMap=torsion_atom_id_map)
        [align_f(Chem.Mol(m2)) for m2 in opt_tp["ROMol"]]

    return opt_tp, merged_energies, vars(tpg)


class LandscaperTorsionProfileGenerator(_AbstractTorsionProfileGenerator):
    """
    Landscaper approach implementation
    """

    def __init__(
            self,
            exploration_iterations: int = 15,
            exploration_convergence: bool = True,
            exploration_convergence_tol: float = 10 ** -5,
            exploration_min_iterations: int = 1,
            exploration_max_iterations=10,
            n_processes: int = 5,
            force_constant: float = 1.0**10,
            max_emin_iter: int = 1000,
            n_measurements: int = 37,
    ):
        """
        This class is a nice simple input generation Variant.

        Parameters
        ----------
        n_measurements : int, optional
            points along the rotation (36 translates to 36 points between -180,180), by default 36
        emin : bool, optional
            remove severe clashes, by default True
        force_constant : float, optional
            constant strength for the position restraints during emin, by default 1.e4
        max_emin_iter : int, optional
            maximal emin steps., by default 1000

        """

        self.emin = True
        self.n_measurements = n_measurements
        self.force_constant = force_constant
        self.max_emin_it = max_emin_iter

        self.exploration_iterations = exploration_iterations
        self.exploration_convergence = exploration_convergence
        self.exploration_convergence_tol = exploration_convergence_tol
        self.exploration_min_iterations = exploration_min_iterations
        self.exploration_max_iterations = exploration_max_iterations

        self.n_processes = n_processes


        self.iteration_min = None
        self.iteration_max = None
        self.iteration_v = None
        self.iteration_profile_mol = None
        self.orig_torsion_atom_ids = None
        self.ff_energies = None
        self.sub_ps = None
        self.iteration_2d_v = None

    def _find_curve_extrema(self, energies: list[float]) -> tuple[float, float]:
        """
            finds extreme points of a given curve.
        Parameters
        ----------
        energies:list[float]
            profile values

        Returns
        -------
        tuple[float, float]
            minima and maxima of a curve
        """
        minima = argrelextrema(energies, np.less, mode="wrap")[0]
        maxima = argrelextrema(energies, np.greater, mode="wrap")[0]
        return minima, maxima

    def _iteration_generate_profile(
            self,
            mol: Chem.Mol,
            torsion_atom_ids: tuple[int, int, int, int],
            angle_range: Iterable[float],
            _additional_pos_res: list[int] = None,
            _additional_torsions: list[tuple[int, int, int, int, float]] = None,
            dev: float = 0.001,
    ) -> pd.DataFrame:
        """
        iterate over profiling

        Parameters
        ----------
        mol: Chem.Mol
        torsion_atom_ids: tuple[int, int, int, int]
        angle_range: Iterable[float]
        _additional_pos_res: list[int]
        _additional_torsions: list[tuple[int, int, int, int, float]]
        dev:float

        Returns
        -------
        result of this iteration

        """

        mols = []
        self.ff_energies = []
        for i, angle in enumerate(angle_range):
            mmol = Chem.Mol(mol)
            conf = mmol.GetConformer()
            conf.SetId(i)
            Chem.rdMolTransforms.SetDihedralDeg(
                conf,
                torsion_atom_ids[0],
                torsion_atom_ids[1],
                torsion_atom_ids[2],
                torsion_atom_ids[3],
                angle,
            )

            if self.emin:
                mp = AllChem.MMFFGetMoleculeProperties(mmol)
                ff = AllChem.MMFFGetMoleculeForceField(mmol, mp)

                ff.MMFFAddTorsionConstraint(
                    torsion_atom_ids[0],
                    torsion_atom_ids[1],
                    torsion_atom_ids[2],
                    torsion_atom_ids[3],
                    False,
                    angle - dev,
                    angle + dev,
                    self.force_constant,
                )
                for a in torsion_atom_ids:
                    ff.MMFFAddPositionConstraint(
                        idx=a, maxDispl=0, forceConstant=self.force_constant
                    )

                if _additional_torsions is not None:
                    for a1, a2, a3, a4, aangle in _additional_torsions:
                        ff.MMFFAddTorsionConstraint(
                            a1, a2, a3, a4, False, aangle - dev, aangle + dev, self.force_constant
                        )

                if _additional_pos_res is not None:
                    for i in _additional_pos_res:
                        ff.MMFFAddPositionConstraint(
                            idx=i, maxDispl=0, forceConstant=self.force_constant
                        )

                ff.Minimize(maxIts=self.max_emin_it)
                v = ff.CalcEnergy()
                self.ff_energies.append(v)

                if(mol.HasProp("molID")):
                    mol_id = mol.GetProp("molID")
                elif(mol.HasProp("_Name")):
                    mol_id = mol.GetProp("_Name")
                else:
                    mol_id = f"mol"

                mol_angle = mmol
                mol_angle.SetProp(key="molID", val=mol_id)
                mol_angle.SetProp(key="torsion_atom_ids", val=str(torsion_atom_ids))
                mol_angle.SetProp(key="torsion_angle", val=str(angle))
                mol_angle.SetProp(key="tpg_mmff94_opt_v", val=str(v))

                if _additional_torsions is not None:
                    for t in _additional_torsions:
                        mol_angle.SetProp(key="torsion_atom_ids_b", val=str(t[:4]))
                        mol_angle.SetProp(key="torsion_angle_b", val=str(t[4]))

                mols.append(mol_angle)

        itp_df = mols_to_moldf(mols)
        itp_df["torsion_angle"] = itp_df["torsion_angle"].round()
        return itp_df

    def _optimize_landscape_from_minima(
            self, init_tp, torsion_atom_ids, angle_range, _additional_pos_res=None, _additional_torsions=None
    ) -> tuple[pd.DataFrame, list[float]]:
        """

        Parameters
        ----------
        init_tp
        torsion_atom_ids
        ff_energies
        angle_range
        _additional_pos_res
        _additional_torsions

        Returns
        -------

        """
        # find minimas
        input_energies = init_tp["tpg_mmff94_opt_v"].to_numpy()
        minima, maxima = self._find_curve_extrema(input_energies)
        self.iteration_min.append(minima)
        self.iteration_max.append(maxima)

        # optimize from minima:
        sub_energies = []
        sub_ps = []
        for mini in minima:
            nm = Chem.Mol(init_tp["ROMol"].iloc[mini])
            tp = self._iteration_generate_profile(
                nm,
                torsion_atom_ids,
                angle_range=angle_range,
                _additional_torsions=_additional_torsions,
                _additional_pos_res=None,
            )
            sub_ps.append(tp)
            sub_energies.append(np.array(self.ff_energies))
        self.sub_ps = sub_ps

        # build final E - merge
        merged_e = input_energies
        rows = [None for _ in range(self.n_measurements)]
        for j in range(self.n_measurements):
            row = init_tp.iloc[j]
            for i, _ in enumerate(minima):
                if merged_e[j] > sub_energies[i][j]:
                    row = sub_ps[i].iloc[j]
                    merged_e[j] = sub_energies[i][j]
            rows[j] = row

        opt_tp = pd.DataFrame(rows)

        return opt_tp, merged_e

    def landscape_exploration(
            self, in_mol, t1, _additional_pos_res=None, _additional_torsions=None, _align_mols: bool = True
    ) -> tuple[Chem.Mol, list[float]]:
        """

        Parameters
        ----------
        in_mol
        t1
        _additional_pos_res
        _additional_torsions
        _align_mols

        Returns
        -------

        """
        angle_range = np.round(
            np.linspace(-180, 180, self.n_measurements, endpoint=True)
        )  # not  a setting currently
        self.iteration_min = []
        self.iteration_max = []
        self.iteration_v = []
        self.iteration_profile_mol = []
        self.orig_torsion_atom_ids = deepcopy(t1)

        # Inital run
        itp_df = self._iteration_generate_profile(
            in_mol,
            t1,
            angle_range=angle_range,
            _additional_pos_res=_additional_pos_res,
            _additional_torsions=_additional_torsions,
        )
        merged_energies = self.ff_energies

        self.iteration_v.append(merged_energies)
        self.iteration_profile_mol.append(itp_df)

        # Explore potential landscape
        for it in range(self.exploration_iterations):
            itp_df, new_merged_energies = self._optimize_landscape_from_minima(
                init_tp=itp_df,
                torsion_atom_ids=t1,
                angle_range=angle_range,
                _additional_torsions=_additional_torsions,
            )

            self.iteration_v.append(new_merged_energies)
            self.iteration_profile_mol.append(itp_df)

            # Convergence Crit
            if (
                    self.exploration_min_iterations <= it
                    and self.exploration_convergence
                    and np.allclose(
                new_merged_energies, merged_energies, rtol=self.exploration_convergence_tol
                )
            ) or self.exploration_max_iterations < it:
                merged_energies = np.array(new_merged_energies)
                log.info("converged in: ", it + 1)
                break

            merged_energies = np.array(new_merged_energies)
            t1 = t1[::-1]  # switch rotation dir for ehanced sampling

        # Final curve extrema
        minima, maxima = self._find_curve_extrema(merged_energies)
        self.iteration_min.append(minima)
        self.iteration_max.append(maxima)

        if _align_mols:
            torsion_atom_id_map = list(zip(t1[:3], t1[:3]))
            align_f = lambda m: Chem.rdMolAlign.AlignMol(prbMol=m, refMol=in_mol,
                                                          atomMap=torsion_atom_id_map)
            [align_f(Chem.Mol(m)) for m in itp_df["ROMol"]]

        itp_df["torsion_atom_ids"] = itp_df["torsion_atom_ids"].apply(lambda x: t1)
        return itp_df, merged_energies

    def landscape_exploration_2d(self, in_mol, t1, t2, _additional_pos_res=None, _align_mols: bool
    = True) -> pd.DataFrame:
        """

        Parameters
        ----------
        in_mol
        t1
        t2
        _additional_pos_res
        _align_mols

        Returns
        -------

        """
        angle_range = np.round(
            np.linspace(-180, 180, self.n_measurements)
        )  # not  a setting currently

        self.orig_torsion_atom_ids = t2
        tp_t2 = self._iteration_generate_profile(
            mol=in_mol, torsion_atom_ids=t2, angle_range=angle_range,
            _additional_pos_res=_additional_pos_res
        )
        ms = tp_t2["ROMol"]

        self.iteration_2d_v = []
        m_opt_p = Chem.Mol(in_mol)
        m_opt_p.RemoveAllConformers()

        with Pool(self.n_processes) as p:
            tasks = [
                (
                    i,
                    m1,
                    t1,
                    _additional_pos_res,
                    [[t2[0], t2[1], t2[2], t2[3], a]],
                    _align_mols,
                    vars(self),
                )
                for i, (a, m1) in enumerate(zip(angle_range, ms))
            ]
            res = p.starmap(
                thread_landscape_exploration_2d,
                tqdm(tasks, total=len(tasks), desc="Calculate landscape (batched): "),
            )
            self.iteration_2d_v = np.array([r[1] for r in res])

        opt_tp = pd.concat([tp for tp,_,_ in res])

        if _align_mols:
            torsion_atom_id_map = list(zip(t1 + t2, t1 + t2))
            align_f = lambda m: Chem.rdMolAlign.AlignMol(prbMol=m, refMol=in_mol,
                                                          atomMap=torsion_atom_id_map)
            [align_f(Chem.Mol(m)) for m in opt_tp["ROMol"]]

        return opt_tp, self.iteration_2d_v

    def generate(
            self,
            mol: Chem.Mol,
            torsion_atom_ids: tuple[int, int, int, int],
            _additional_pos_res: list[int] = None,
            _additional_torsions: list[tuple[int, int, int, int, float]] = None,
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        mol
        torsion_atom_ids
        _additional_pos_res
        _additional_torsions

        Returns
        -------

        """
        c_mol = Chem.Mol(mol)
        if c_mol.GetNumConformers() == 0:
            Chem.rdDistGeom.EmbedMolecule(c_mol)  # if no coords, guess

        tp, v = self.landscape_exploration(
            c_mol,
            torsion_atom_ids,
            _additional_pos_res=_additional_pos_res,
            _additional_torsions=_additional_torsions,
        )
        self.ff_energies = np.array(v)

        if mol.HasProp("_Name"):
            mol_name = mol.GetProp("_Name")
            tp["_Name"] = mol_name
        return tp

    def generate_2d(
            self,
            mol: Chem.Mol,
            torsion_atom_ids1: tuple[int, int, int, int],
            torsion_atom_ids2: tuple[int, int, int, int],
            _additional_pos_res: list[int] = None,
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        mol
        torsion_atom_ids1
        torsion_atom_ids2
        _additional_pos_res

        Returns
        -------

        """
        tp, V = self.landscape_exploration_2d(
            mol, torsion_atom_ids1, torsion_atom_ids2, _additional_pos_res=_additional_pos_res
        )
        self.ff_energies = np.array(V)

        if mol.HasProp("_Name"):
            mol_name = mol.GetProp("_Name")
            tp["molID"] = mol_name

        return tp
