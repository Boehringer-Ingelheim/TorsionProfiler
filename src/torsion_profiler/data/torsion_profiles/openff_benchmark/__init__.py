"""
OpenFF Industry Benchmark mols
"""
import warnings
from collections import defaultdict

from rdkit import Chem

from ....utils import bash
from ....utils.utils import RingError

root_path = bash.path.dirname(__file__)


class OpenFFGen2TorsionSetbase:
    """
    Builder extracts rotatable torsions of the OpenFF Benchmark set by:
    TODO: add citation

    """
    data_set_1_path: str
    data_set_2_path: str

    @staticmethod
    def sort_tors_ids(mol, tors_id):
        tors_atoms = [mol.GetAtomWithIdx(i) for i in tors_id]

        atom_bond_partner = defaultdict(list)
        for a in tors_atoms:
            for b in a.GetBonds():
                if b.GetBeginAtomIdx() != a.GetIdx():
                    atom_bond_partner[a.GetIdx()].append(b.GetBeginAtomIdx())
                else:
                    atom_bond_partner[a.GetIdx()].append(b.GetEndAtomIdx())

        i_l = []
        j_k = []
        for a, bond_atoms in atom_bond_partner.items():
            count = sum(1 for i in tors_id if i in bond_atoms)
            if count == 2:
                j_k.append(a)
            else:
                i_l.append(a)

        tors_core = list(sorted(j_k))
        for a, bond_atoms in filter(lambda x: not x[0] in tors_core, atom_bond_partner.items()):
            if tors_core[0] in bond_atoms:
                i = a
            elif tors_core[1] in bond_atoms:
                l = a
            else:
                return tors_id

        torsion_atom_ids = [i] + j_k + [l]
        return torsion_atom_ids

    @classmethod
    def _get_h_atom(cls, a1:Chem.Atom)->int:
        """
        get a hatom adjacent to a1

        Parameters
        ----------
        a1: Chem.Atom

        Returns
        -------
        int
            atom id
        """
        new_id = None

        for b in a1.GetBonds():
            ba = b.GetBeginAtom()
            bb = b.GetEndAtom()

            if ba != a1 and ba.GetSymbol() == "H":
                new_id = ba.GetIdx()
                break

            if bb != a1 and bb.GetSymbol() == "H":
                new_id = bb.GetIdx()
                break

        return new_id

    @classmethod
    def _middle_atom(cls, tors_atom: Chem.Atom, torsion_atoms: list[Chem.Atom]) -> bool:
        """
        is the given atom a middle atom in the torsion?

        Parameters
        ----------
        tors_atom: Chem.Atom
        torsion_atoms: list[Chem.Atom]

        Returns
        -------
        bool

        """
        atom_bonds = tors_atom.GetBonds()
        torsino_atoms = [t.GetIdx() for t in torsion_atoms]
        bond_atoms = []
        for b in atom_bonds:
            bond_atoms.extend([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
        bond_atoms = set(bond_atoms)

        count = sum(1 for i in torsino_atoms if i in bond_atoms) - 1

        if count == 1:
            return False
        if count == 2:
            return True
        raise ValueError("This atom is strange! Is connected to torsion atoms: " + str(count))

    @classmethod
    def _get_tors_atom_ids(cls, mol: Chem.Mol)->list[int]:
        """
        get torsion atoms of a mol

        Parameters
        ----------
        mol: Chem.Mol

        Returns
        -------
        list[int]
            torsion atom ids
        """
        real_torsion_atom_ids = {}
        atoms = []
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            aid = atom.GetIdx()
            if int(map_num) > 0:
                real_torsion_atom_ids[map_num] = aid
                atoms.append(atom)
        real_torsion_atom_ids = [real_torsion_atom_ids[k] for k in sorted(real_torsion_atom_ids)]

        # Fix Hs
        if len(real_torsion_atom_ids) < 4:
            if len(real_torsion_atom_ids) == 2:
                real_torsion_atom_ids = (
                    [cls._get_h_atom(atoms[0])]
                    + real_torsion_atom_ids
                    + [cls._get_h_atom(atoms[1])]
                )

            elif len(real_torsion_atom_ids) == 3:
                t = (
                    cls._get_h_atom(atoms[2])
                    if (not cls._middle_atom(atoms[2], atoms))
                    else None
                )
                t2 = (
                    cls._get_h_atom(atoms[0])
                    if (not cls._middle_atom(atoms[0], atoms))
                    else None
                )
                t3 = (
                    cls._get_h_atom(atoms[1])
                    if (not cls._middle_atom(atoms[1], atoms))
                    else None
                )

                if t is not None:
                    real_torsion_atom_ids = real_torsion_atom_ids + [t]
                elif t2 is not None:
                    real_torsion_atom_ids = [t2] + real_torsion_atom_ids
                elif t3 is not None:
                    real_torsion_atom_ids = [t3] + real_torsion_atom_ids
                else:
                    raise RingError("did not find a H for three ids!")
        return real_torsion_atom_ids

    @classmethod
    def _prepare_mols(cls, suppl:Chem.SDMolSupplier)->list[Chem.Mol]:
        """
            prepare mols and parse accordint to our scheme

        Parameters
        ----------
        suppl:Chem.SDMolSupplier

        Returns
        -------
        list[Chem.Mol]
            molecules
        """
        mols = list(suppl)
        ring_err = 0
        ring_ids = []

        for i, mol in enumerate(mols):
            try:
                torsion_atom_ids = cls._get_tors_atom_ids(mol)
                torsion_atom_ids = cls.sort_tors_ids(mol, torsion_atom_ids)
                # print(i, real_torsion_atom_ids)
                mol.SetProp("torsion_atom_ids", str(torsion_atom_ids))
            except RingError:
                # warnings.warn("this mole torsion is in a ring: "+str(i))
                ring_ids.append(i)
                ring_err += 1
                mol.SetProp("torsion_atom_ids", str(None))
                continue
            except Exception as err:
                raise ValueError(f"{i}\t{mol}\n") from err

        warnings.warn(
            "Total Molecules: "
            + str(len(mols))
            + "\t RingMolecules-skipped: "
            + str(ring_err)
            + "\t"
            + str(ring_ids)
        )
        return mols

    @classmethod
    def get_set_1_mols(cls)->list[Chem.Mol]:
        """
            get first set of molecules

        Returns
        -------
        list[Chem.Mol]
            molecules
        """
        suppl = Chem.SDMolSupplier(cls.data_set_2_path, removeHs=False)
        mols = cls._prepare_mols(suppl)

        return mols

    @classmethod
    def get_set_2_mols(cls)->list[Chem.Mol]:
        """
            get second set of molecules

        Returns
        -------
        list[Chem.Mol]
            molecules
        """
        suppl = Chem.SDMolSupplier(cls.data_set_2_path, removeHs=False)
        mols = cls._prepare_mols(suppl)

        return mols


class OpenFFGen2TorsionSet1Roche(OpenFFGen2TorsionSetbase):
    """
    Class for the Roche Contribution
    """
    data_set_2_path = root_path + "/OpenFF_Gen_2_Torsion_Set_1_Roche_2.sdf"
    data_set_2_path = root_path + "/OpenFF_Gen_2_Torsion_Set_1_Roche.sdf"


class OpenFFGen2TorsionSet2Coverage(OpenFFGen2TorsionSetbase):
    """
    Class for the Coverage Data
    """
    data_set_2_path = root_path + "/OpenFF_Gen_2_Torsion_Set_2_Coverage_2.sdf"
    data_set_1_path = root_path + "/OpenFF_Gen_2_Torsion_Set_2_Coverage.sdf"


class OpenFFGen2TorsionSet3Pfizer(OpenFFGen2TorsionSetbase):
    """
    Class for the Pfizer Contribution
    """
    data_set_2_path = root_path + "/OpenFF_Gen_2_Torsion_Set_3_Pfizer_Discrepancy_2.sdf"
    data_set_1_path = root_path + "/OpenFF_Gen_2_Torsion_Set_3_Pfizer_Discrepancy.sdf"


class OpenFFGen2TorsionSet4eMolecule(OpenFFGen2TorsionSetbase):
    """
    Class for the Discrepancies Data
    """
    data_set_2_path = root_path + "/OpenFF_Gen_2_Torsion_Set_4_eMolecules_Discrepancy_2.sdf"
    data_set_1_path = root_path + "/OpenFF_Gen_2_Torsion_Set_4_eMolecules_Discrepancy.sdf"


class OpenFFGen2TorsionSet5Bayer(OpenFFGen2TorsionSetbase):
    """
    Class for the Bayer Contribution
    """
    data_set_2_path = root_path + "/OpenFF_Gen_2_Torsion_Set_5_Bayer_2.sdf"
    data_set_1_path = root_path + "/OpenFF_Gen_2_Torsion_Set_5_Bayer.sdf"


class OpenFFGen2TorsionSet6Supplemental(OpenFFGen2TorsionSetbase):
    """
    Class for the Supplemental data
    """
    data_set_2_path = root_path + "/OpenFF_Gen_2_Torsion_Set_6_Supplemental_2.sdf"
    data_set_1_path = root_path + "/OpenFF_Gen_2_Torsion_Set_6_Supplemental.sdf"


class OpenFFGen3TorsionSet(OpenFFGen2TorsionSetbase):
    """
    Class for the Gen3 Data
    """
    data_set_2_path = root_path + "/OpenFF_Gen3_Torsion_Set_v1-0.sdf"

    @classmethod
    def get_set_2_mols(cls):
        """
        second set is not avail for this data set.
        """
        raise Exception("not available")


class OpenFFFullDataSet:
    """
     get All gathered OpenFF Data
    """
    @classmethod
    def get_mols(cls)->list[Chem.Mol]:
        """
            get all mols
        Returns
        -------
        list[Chem.Mol]
            molecules
        """
        # Dataset
        mols = []
        for dset in (
            OpenFFGen2TorsionSet1Roche,
            OpenFFGen2TorsionSet2Coverage,
            OpenFFGen2TorsionSet3Pfizer,
            OpenFFGen2TorsionSet4eMolecule,
            OpenFFGen2TorsionSet5Bayer,
            OpenFFGen2TorsionSet6Supplemental,
            OpenFFGen3TorsionSet,
        ):
            mols.extend(dset.get_set_1_mols())
            try:
                mols.extend(dset.get_set_2_mols())
            except Exception:
                pass

        # filter_ring_mols:
        ring_mols = [
            551,
            983,
            984,
            1055,
            1056,
            1057,
            1058,
            804,
            771,
            772,
            913,
            914,
            915,
            916,
            917,
            988,
            989,
            55,
            556,
            557,
            1065,
            668,
            669,
            560,
            561,
            778,
            779,
            780,
            781,
            564,
            565,
            632,
            574,
            507,
            508,
            509,
            929,
            930,
            700,
            701,
            741,
            1032,
            1033,
            706,
            707,
            525,
            526,
            527,
            1199,
            1200,
            1201,
            1202,
            1203,
            1204,
            1206,
            1207,
            1208,
            1209,
            893,
            531,
            532,
            533,
            534,
            535,
            536,
            537,
            538,
            539,
            719,
            720,
            721,
            722,
            723,
            1039,
            1040,
            1041,
            1042,
            1002,
            727,
            728,
            1176,
            1177,
            961,
            962,
            793,
            794,
            795,
            796,
            797,
        ]
        ring_mols = []
        mols = [m for i, m in mols if i not in ring_mols]
        return mols
