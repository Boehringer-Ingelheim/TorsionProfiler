"""
Torsionnet data
"""
import warnings
from rdkit import Chem

from ....utils import bash

root_path = bash.path.dirname(__file__)


class TorsionNetSet:
    """
    TorsionNetSet data set.
    from the publication by:
        TorsionNet: A Deep Neural Network to Rapidly Predict Small-Molecule Torsional Energy
        Profiles with the Accuracy of Quantum Mechanics
        Brajesh K. Rai*, Vishnu Sresht, Qingyi Yang, Ray Unwalla, Meihua Tu, Alan M. Mathiowetz,
        and Gregory A. Bakken
        J. Chem. Inf. Model 2022
    """
    torsionNet_test_data_path = root_path + "/TorsionNet500_mm_opt_geometries.sdf"

    @classmethod
    def get_mols(cls)->list[Chem.Mol]:
        """
        Get the torsionset data

        Returns
        -------
        list[Chem.Mol]
            molecules
        """
        suppl = Chem.SDMolSupplier(cls.torsionNet_test_data_path, removeHs=False)
        warnings.warn(
            "This dataset contains 500 molecules and 24 data points per molecule stored "
            "as seperate mol."
        )
        mols = list(suppl)
        torsion_lists = [
            tuple(map(lambda x: int(x) - 1, mol.GetProp("TORSION_ATOMS_FRAGMENT").split()))
            for mol in mols
        ]

        for m, t in zip(mols, torsion_lists):
            m.SetProp("torsion_atom_ids", str(t))

        return mols
