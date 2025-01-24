"""
abstract calculater baseclass
"""
import abc

import pandas as pd
import numpy as np
from rdkit import Chem

from typing import Iterable
from ..utils.baseclass import BaseClass
from ..utils.metrics import get_boltzman_ensemble_p
from ..utils.units import kcal_to_kJ


class _AbstractCalculator(BaseClass):
    """
    This is a Baseclass for the Calculators
    """
    _conda_env_path: str
    _optimize_structure: bool
    _check_information: dict

    def __init__(self):
        self._optimize_structure = None
        self._optimize_structure_nsteps = None
        self._optimize_structure_tol = None
        self._optimize_structure_write_out = None

    @property
    def optimize_structure(self) -> bool:
        """
        getter optimization flag
        """
        return self._optimize_structure

    @property
    def optimize_structure_nsteps(self) -> int:
        """
        getter optimization number of steps
        """
        return self._optimize_structure_nsteps

    @property
    def optimize_structure_tol(self) -> float:
        """
        getter optimization convergence tolerance
        """
        return self._optimize_structure_tol

    @property
    def optimize_structure_write_out(self) -> bool:
        """
        getter  write out structure
        """
        return self._optimize_structure_write_out

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                map(
                    lambda x: (
                        str(x[0]) + "=" + str(x[1].__name__)
                        if (hasattr(x[1], "__name__"))
                        else str(x[0]) + "=" + str(x[1])
                    ),
                    vars(self).items(),
                )
            )
            + ")"
        )

    @abc.abstractmethod
    def calculate_conformer_potentials(
        self,
        mol: Chem.Mol,
        torsion_atom_ids: tuple[int, int,int, int] = None,
        n_measurements: int = 25,
        out_file: str = None,
        _additional_torsions: list[tuple[int, int, int, int, float]] = None,
    ) -> pd.DataFrame:
        """
        basic function for calculating conformer potentials.
        Parameters
        ----------
        mol: Chem.Mol
        torsion_atom_ids: tuple[int, int,int, int]
        n_measurements: int
        out_file: str
        _additional_torsions: list[tuple[int, int, int, int, float]]

        Returns
        -------
        pd.DataFrame
            results form the optimization.
        """
        raise NotImplementedError()

    def calculatePotential(
        mol: Chem.Mol,
        torsion_atom_ids: tuple[int, int,int, int],
        n_measurements: int = 24,
        out_file: str = None,
        _additional_torsion: tuple[int, int, int, int, float] = None,
    ) -> pd.DataFrame:
        """
            base function calculating the potential.

        Parameters
        ----------
        mol: Chem.Mol
        torsion_atom_ids: tuple[int, int,int, int]
        n_measurements: int
        out_file: str
        _additional_torsions: list[tuple[int, int, int, int, float]]

        Returns
        -------
        pd.DataFrame
            results form the optimization.
        """
        raise NotImplementedError()

    @property
    def name(self) -> str:
        """
        getter for name of Calculator.
        """
        return self.__class__.__name__.replace("_calculator", "")
