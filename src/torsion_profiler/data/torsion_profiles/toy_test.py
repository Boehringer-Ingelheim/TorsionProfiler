"""
Implment ToyTestSet dataset
"""
import itertools as it
from rdkit import Chem


class ToyTestSet:
    """
    Toy data set.
    """
    @classmethod
    def get_all_butan_comibs(cls, elements: tuple[str]=("C", "N", "O", "S"))->list[Chem.Mol]:
        """
        generates all 4 atom molecules out of given elements

        Parameters
        ----------
        elements: tuple[str]

        Returns
        -------
        list[Chem.Mol]
            generated molecules
        """
        all_orders = set(list(it.combinations_with_replacement(elements, 4)))
        all_combs = [list(it.permutations(x)) for x in all_orders]

        fin = []
        for l in all_combs:
            fin.extend(l)
        all_possible_combinations = sorted(set(fin))

        mols = list(map(lambda x: Chem.MolFromSmiles("".join(x)), all_possible_combinations))
        mols = [Chem.AddHs(m) for m in mols]

        for m in mols:
            Chem.rdDistGeom.EmbedMolecule(m)

        return mols

    @classmethod
    def get_reasonable_mols(cls)-> list[Chem.Mol]:
        """
            generate the given set of 4 atom molecules

        Returns
        -------
        list[Chem.Mol]
            generated molecules
        """
        smis = ["CCCC", "CCCN", "CCCO", "OCCN", "NCCN", "OCCO", "CCOC", "CCNC", "CCCS", "SCCS"]
        torsion_atom_ids = (0, 1, 2, 3)

        mols = list(map(lambda x: Chem.MolFromSmiles("".join(x)), smis))
        mols = [Chem.AddHs(m) for m in mols]

        for m in mols:
            m.SetProp("torsion_atom_ids", str(torsion_atom_ids))
            Chem.rdDistGeom.EmbedMolecule(m)

        return mols

    @classmethod
    def get_mols(cls) -> list[Chem.Mol]:
        """
                    generate the given set of 4 atom molecules

        Returns
        -------
         list[Chem.Mol]
            generated mols
        """
        return cls.get_reasonable_mols()
