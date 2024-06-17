"""
Fast profile generator generating input torsion profile
"""

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetDihedralDeg

from ....utils import mols_to_moldf
from ._abstract_torsion_profile_generator import _AbstractTorsionProfileGenerator

log = logging.getLogger(__name__)


def periodic_angle(angle: float) -> float:
    """
    this is a function for fixing periodc angle problem.
    Place each angle between -180, 180

    Parameters
    ----------
    angle: float

    Returns
    -------
    float
        angle brought to range -180, 180
    """
    if angle > 180:
        rangle = angle % 360
        rangle -= 360
        if rangle < -180:
            rangle = rangle % 180
    elif angle < -180:
        rangle = angle % -360
        rangle += 360
        if rangle > 180:
            rangle = rangle % -180
    else:
        rangle = angle
    return rangle


class FastProfileGenerator(_AbstractTorsionProfileGenerator):
    """
    FastProfileGenerator implementation
    """
    def __init__(
        self,
        emin: bool = True,
        force_constant: float = 1.0e4,
        max_emin_iter: int = 1000,
        n_measurements: int = 37,
    ):
        """
        This class is a very simple input generation Variant.

        Parameters
        ----------
        n_measurements : int, optional
            points along the rotation (36 translates to 36 points between -180,180), by default 36
        emin : bool, optional
            remove severe clashes, by default True
        force_constant : float, optional
            constant strength for the position restraints during emin, by default 1.e4
        maxEminIt : int, optional
            maximal emin steps., by default 1000

        """

        self.emin = emin
        self.n_measurements = n_measurements
        self.force_constant = force_constant
        self.max_emin_it = max_emin_iter
        self.ff_energies = None

    def generate(
        self,
        mol: Chem.Mol,
        torsion_atom_ids: tuple[int, int, int, int],
        _additional_pos_res: list[int] = None,
        _additional_torsions: list[tuple[int, int, int, int, float]] = None,
    ) -> pd.DataFrame:
        """
        This function rotates a torsion dihedral by 360 degrees and removes severe clashes
        if emin is True with an MM Force field.

        Parameters
        ----------
        mol : Chem.Mol
            molecule of interest
        torsion_atom_ids : list[int]
            torsion atom ids, that shall be rotated
        n_measurements : int, optional
            points along the rotation (36 translates to 36 points between -180,180), by default 36
        emin : bool, optional
            remove severe clashes, by default True
        force_constant : float, optional
            constant strength for the position restraints during emin, by default 1.e4
        maxEminIt : int, optional
            maximal emin steps., by default 1000

        Returns
        -------
        Chem.Mol
            molecule containing the torsion profile structures as conformers.
        """

        angle_range = np.round(
            np.linspace(
                0,
                360,
                self.n_measurements,
                endpoint=True,
            )
        )
        c_mol = Chem.Mol(Chem.AddHs(mol, addCoords=True))

        if c_mol.GetNumConformers() == 0:
            Chem.rdDistGeom.EmbedMolecule(c_mol)  # if no coords, guess

        try:
            itp_df = self._minimize_rotation(
                c_mol,
                torsion_atom_ids[::-1],
                angle_range,
                _additional_pos_res,
                _additional_torsions,
                dev=5,
            )
        except RuntimeError:
            raise RuntimeError
            log.warning("Caught a Runtime Err, will try reverse torsions!")
            itp_df = self._minimize_rotation(
                c_mol,
                torsion_atom_ids[::1],
                angle_range,
                _additional_pos_res,
                _additional_torsions,
                dev=5,
            )

        # nice align
        torsion_atom_id_map = list(zip(torsion_atom_ids, torsion_atom_ids))
        align_f = lambda m: Chem.rdMolAlign.AlignMol(prbMol=m, refMol=c_mol,
                                                     atomMap=torsion_atom_id_map)
        [align_f(Chem.Mol(m)) for m in itp_df["ROMol"]]

        return itp_df

    def _minimize_rotation(
        self, mol:Chem.Mol, torsion_atom_ids: tuple[int,int,int,int],
            angle_range: Iterable[float], _additional_pos_res:list[int],
            _additional_torsions: list[tuple[int, int, int, int, float]] , dev:float=0.01
    ) -> pd.DataFrame:
        """
        minimize one cofnormer - helper function

        Parameters
        ----------
        mol:Chem.Mol
        torsion_atom_ids: tuple[int,int,int,int]
        angle_range: Iterable[float]
        _additional_pos_res: list[int]
        _additional_torsions:list[tuple[int, int, int, int, float]]
        dev: float

        Returns
        -------
        Chem.Mol
            result profile
        """
        tmol = Chem.Mol(mol)
        tmol.RemoveAllConformers()

        mmol = Chem.Mol(mol)
        conf = mmol.GetConformer()

        # round to 10th number
        rangle = (
            np.round(
                Chem.rdMolTransforms.GetDihedralDeg(
                    conf,
                    torsion_atom_ids[0],
                    torsion_atom_ids[1],
                    torsion_atom_ids[2],
                    torsion_atom_ids[3],
                )
                / 10
            )
            * 10
        )

        mols = []
        self.ff_energies = []
        for i, dangle in enumerate(angle_range):
            angle = periodic_angle(rangle + dangle)

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

                mol_angle = Chem.Mol(tmol)
                mol_angle.AddConformer(conf)
                mol_angle.SetProp(key="torsion_atom_ids", val=str(torsion_atom_ids))
                mol_angle.SetProp(key="torsion_angle", val=str(angle))
                mol_angle.SetProp(key="tpg_mmff94_opt_v", val=str(v))
                mols.append(mol_angle)

        itp_df = mols_to_moldf(mols)

        return itp_df
