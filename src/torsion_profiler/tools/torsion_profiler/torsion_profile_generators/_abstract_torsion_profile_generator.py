"""
Abstract class on how a torsion profile generator should work.
"""
import abc

from rdkit import Chem


class _AbstractTorsionProfileGenerator(abc.ABC):
    """
    Torsion profile generator base interface.
    """
    @abc.abstractmethod
    def generate(
        self,
        mol: Chem.Mol,
        torsion_atom_ids: tuple[int, int, int, int],
        _additional_pos_res: list[int] = None,
        _additional_torsions: list[tuple[int, int, int, int, float]] = None,
    ) -> Chem.Mol:
        """
            generate an initial torsion profile.

        Parameters
        ----------
        mol: Chem.Mol
            start conformer
        torsion_atom_ids: tuple[int, int, int, int]
            atom ids defining a torsion
        _additional_pos_res: list[int]
            atom ids for position restraints
        _additional_torsions: list[tuple[int, int, int, int, float]]
            additional torsion restraints

        Returns
        -------
        Chem.Mol
            initial profile mol

        """
