"""
BiAryl data
"""

from rdkit import Chem

from ....utils import bash

root_path = bash.path.dirname(__file__)


class RowleyBiarylSet:
    """
    Here we provide data from:
    Benchmarking Force Field and the ANI Neural Network Potentials for the Torsional Potential
    Energy Surface of Biaryl Drug Fragments
    Shae-Lynn J. Lahey, Tu Nguyen Thien Phuc, and Christopher N. Rowley*
    J. Chem. Inf. Model. 2020
    """

    torsionNet_test_data_path = root_path + "/OpenFF_Rowley_Biaryl_v1-0.sdf"

    @classmethod
    def get_mols(cls):
        """
        Get the BiAryl data

        Returns
        -------
        list[Chem.Mol]
            molecules
        """
        suppl = Chem.SDMolSupplier(cls.torsionNet_test_data_path, removeHs=False)
        mols = list(suppl)
        for i, mol in enumerate(mols):
            real_torsion_atom_ids = []
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                aid = atom.GetIdx()
                if int(map_num) > 0:
                    real_torsion_atom_ids.append(aid)
            mol.SetProp("torsion_atom_ids", str(real_torsion_atom_ids))
            mol.SetProp("molID", "Rowely_mol_" + str(i))

        return mols
