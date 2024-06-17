"""
MMFF94 Calculator implementation
"""
import logging
from typing import Iterable, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from ..utils import store_mol_db, mols_to_moldf
from ._abstractcalculator import _AbstractCalculator

log = logging.getLogger(__name__)
class Mmff94Calculator(_AbstractCalculator):
    # Coordinate Optimization:
    _optimize_structure: bool
    _optimize_structure_nsteps: int
    _optimize_structure_tol: float
    _optimize_structure_write_out: bool

    # Magics
    def __init__(
        self,
        ff_model: str = "MMFF94",
        optimize_structure: bool = True,
        optimize_structure_nsteps: int = 100000,
        optimize_structure_tol: float = 10**-6,
        optimize_structure_write_out: bool = True,
        force_constant: float = 10**4,
    ) -> None:
        """
        Construct MMFF94 Calculator
        Parameters
        ----------
        ff_model: str
            force field models
        optimize_structure: bool
            optimize structure
        optimize_structure_nsteps: int
            number of optimization steps
        optimize_structure_tol: float
            convergence tolerance
        optimize_structure_write_out: bool
            write out structure files?
        force_constant: float
            force_constant for restraints
        """
        super().__init__()

        self.ff_model = ff_model
        self._optimize_structure = optimize_structure
        self._optimize_structure_nsteps = optimize_structure_nsteps
        self._optimize_structure_tol = optimize_structure_tol
        self._optimize_structure_write_out = optimize_structure_write_out
        self.force_constant = force_constant

    def _calculate_optimizations_with_rdkit(
        self,
        mmol: Iterable[Chem.Mol],
        torsion_atom_ids: tuple[int, int, int, int] = None,
        _conf_range: Iterable[int] = None,
        _additional_pos_res: list[int] = None,
        _additional_torsions: list[tuple[int, int, int, int, float]] = None,
    )-> pd.DataFrame:
        """
        calculate the potentials.

        Parameters
        ----------
        mol: Chem.Mol
            molecule of desire
        torsion_atom_ids: tuple[int, int, int, int]
            atom ids defining the torsion
        _conf_range: Iterable[int]
            conformers to optimize
        _additional_pos_res: list[int]
            additional position restraints for atom ids
        _additional_torsions: list[tuple[int, int, int, int, float]]
            additional torsion restraints

        Returns
        -------
        dict[int, float]
            conformer - result energy
        """
        dev = 0.1

        out_mols = []
        for i in _conf_range:
            out_mol = Chem.Mol(mmol[i])
            conf = out_mol.GetConformer()
            mp = AllChem.MMFFGetMoleculeProperties(out_mol)
            ff = AllChem.MMFFGetMoleculeForceField(out_mol, mp)

            if torsion_atom_ids is not None:
                if not self._optimize_structure:
                    log.warning(
                        "No optimization - therefore the restraints force constant is "
                        "changed to 10**-3"
                    )
                    self.force_constant = 10**-3.8

                angle = Chem.rdMolTransforms.GetDihedralDeg(
                    conf,
                    torsion_atom_ids[0],
                    torsion_atom_ids[1],
                    torsion_atom_ids[2],
                    torsion_atom_ids[3],
                )

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

                out_mol.SetProp(key="torsion_atom_ids", val=" ".join(map(str, torsion_atom_ids)))
                out_mol.SetProp(key="torsion_angle", val=str(angle))
                out_mol.SetProp("confID", str(i))

            else:
                out_mol.SetProp("calculationID", str(i))

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

            if self.optimize_structure:
                ff.Minimize(
                    maxIts=self._optimize_structure_nsteps, energyTol=self._optimize_structure_tol
                )

            v = ff.CalcEnergy()
            out_mol.SetProp("potential_energy", str(v))
            out_mols.append(out_mol)

        mol_db = mols_to_moldf(out_mols)

        return mol_db

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

        if(isinstance(mol, Chem.Mol)):
            calculate_mols = [mol]
        elif(isinstance(mol, pd.DataFrame)):
            calculate_mols = mol["ROMol"].to_list()
        elif (isinstance(mol, pd.DataFrame)):
            calculate_mols = mol
        else:
            raise IOError("Need to have an acceptable input type")

        if _conf_range is None:
            _conf_range = range(len(calculate_mols))


        # Calculate / if no optimize, the _optimize_structure attribute will control this
        mol_db = self._calculate_optimizations_with_rdkit(
            mmol=calculate_mols,
            torsion_atom_ids=torsion_atom_ids,
            _conf_range=_conf_range,
            _additional_torsions=_additional_torsions,
        )

        if out_file_path is not None:
            store_mol_db(mol_db, out_file_path, )

        return mol_db
