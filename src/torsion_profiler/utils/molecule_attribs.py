"""
Functions for calculating molecule attributes.
"""

from rdkit import Chem

from openff.fragmenter.fragment import PfizerFragmenter, Fragmenter
from openff.toolkit.topology import Molecule

from openff.fragmenter.utils import (
    get_atom_index,
    get_map_index,
)


def get_rotatable_bonds(mol: Chem.Mol) -> list[Chem.Bond]:
    """Function to find all rotatable bonds in a molecule taking symmetry into account.

    Parameters
    ----------
    mol : Chem.Mol
        The rdkit.Chem.Mol object

    Returns
    -------
    list[Chem.Bond]
         list of rdkit.Chem.Bond that were found in a molecule taking symmetry into account.
    """
    rotatable_bond_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")

    def find_rbs(x, y=rotatable_bond_smarts):
        return x.GetSubstructMatches(y, uniquify=1)

    rbs = find_rbs(mol)
    bonds = [mol.GetBondBetweenAtoms(*inds) for inds in rbs]

    return bonds


def get_torsion_atoms_idx(bond: Chem.Bond) -> list[int]:
    """Function that finds the atomic ids that specify a torsion around the bond of interest.

    Parameters
    ----------
    bond : Chem.Bond
        The bond of interest around which the torsion should be specified.

    Returns
    -------
    list[int]
        list of atomic ids that specify a torsion around the bond of interest.
    """
    bond_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
    additional_atom1 = list(
        filter(lambda x: x.GetIdx() != bond_atoms[1].GetIdx(), bond_atoms[0].GetNeighbors())
    )[0]
    additional_atom2 = list(
        filter(lambda x: x.GetIdx() != bond_atoms[0].GetIdx(), bond_atoms[1].GetNeighbors())
    )[0]
    torsion_atoms = [additional_atom1] + list(bond_atoms) + [additional_atom2]
    torsion_atom_ids = [a.GetIdx() for a in torsion_atoms]

    return torsion_atom_ids


def fragment_mol_complete(
    mol: Chem.Mol, verbose: bool = True, fragmenter_engine:Fragmenter=PfizerFragmenter()
)->dict[str, Chem.Mol]:
    """
    this function fragments the given input molecule.

    Parameters
    ----------
    mol: Chem.Mol
    verbose: bool
    fragmenter_engine: openff_fragmenter.Fragmenter


    Returns
    -------
    dict[str, Chem.Mol]
        fragmented mols
    """
    querry_mol = Molecule.from_rdkit(mol)

    # Calculate Fragments
    if verbose:
        print("fragment")
    result = fragmenter_engine.fragment(querry_mol)

    # build up tasks
    fragment_calcs = []
    fragment_target_bond = []
    for i, fragment in enumerate(result.fragments):
        if verbose:
            print("fragment", i, "getFrag")
        fragment_mol = fragment.molecule
        original_target_bond = fragment.bond_indices
        fragment_target_bond = [
            get_atom_index(fragment_mol, atom_id) for atom_id in original_target_bond
        ]

        # build up torsion dict
        if verbose:
            print("fragment", i, "Tors")
        frag_torsion_atom_ids_dict = dict(zip((2, 3), fragment_target_bond))
        for atom_id in sorted(fragment_target_bond):
            a = fragment_mol.atoms[atom_id]

            bondeds = list(a.bonded_atoms)
            heavier = sorted(bondeds, key=lambda x: x.name == "H")
            not_bond = list(
                filter(lambda x: not x.molecule_atom_index in fragment_target_bond, heavier)
            )
            possible_atom_ids = [pa.molecule_atom_index for pa in not_bond]

            if 1 in frag_torsion_atom_ids_dict:
                frag_torsion_atom_ids_dict[4] = possible_atom_ids[0]
            else:
                frag_torsion_atom_ids_dict[1] = possible_atom_ids[0]

        # Mapping big to big mol
        torsion_i_ds_whole_mol_to_frag_fol = {
            a: get_atom_index(result.parent_molecule, get_map_index(fragment_mol, a))
            for i, a in frag_torsion_atom_ids_dict.items()
        }

        # prep Rdkit Mol
        if verbose:
            print("fragment", i, "RDKIT")
        fragment_rdmol = fragment_mol.to_rdkit()
        for i, v in frag_torsion_atom_ids_dict.items():
            a = fragment_rdmol.GetAtomWithIdx(v)
            a.SetAtomMapNum(i)
        frag_torsion_atom_ids = list(
            map(lambda x: x[1], sorted(frag_torsion_atom_ids_dict.items()))
        )
        fragment_rdmol.SetProp("torsion_atom_ids", str(frag_torsion_atom_ids))

        fragment_calcs.append(
            (
                fragment_rdmol,
                tuple(frag_torsion_atom_ids),
                tuple(torsion_i_ds_whole_mol_to_frag_fol),
                result.parent_molecule.to_rdkit(),
            )
        )

    return fragment_calcs


def filter_rotatable_bonds(rotatable_bonds: list[Chem.Bond]) -> list[Chem.Bond]:
    """The function to filter out the torsions about such bonds as: terminal methyls,
    primary amines, alcohols, and halogens

    Parameters
    ----------
    rotatable_bonds : list[Chem.Bond]
        list of rdkit.Chem.Bond that were found with function get_rotatable_bonds()

    Returns
    -------
    list[Chem.Bond]
        list of rdkit.Chem.Bond that were filtered. The list doesn't contain any bonds to terminal
        methyls, primary amines, alcohols, and halogens.
    """
    filtered_bonds = []

    # Atomic numbers for H,F,Cl,Br,I
    single_valence_atoms = [1, 9, 17, 35, 53]

    for bond in rotatable_bonds:
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        begin_atom_number = begin_atom.GetAtomicNum()
        end_atom_number = end_atom.GetAtomicNum()

        begin_atom_neighbors = begin_atom.GetNeighbors()
        end_atom_neighbors = end_atom.GetNeighbors()

        # check begin atom to be methyl or substituted methyl
        if begin_atom_number == 6:
            if len(begin_atom_neighbors) == 4:
                acc = []
                for neighbor in begin_atom_neighbors:
                    neighbor_is_single_valenced = neighbor.GetAtomicNum() in single_valence_atoms
                    if neighbor_is_single_valenced:
                        acc.append(neighbor_is_single_valenced)

                total_num_single_valenced_neighb = len(acc)

                if total_num_single_valenced_neighb >= 3:
                    continue
        # check begin atom to be OH or NH2
        if begin_atom_number in [8, 7]:
            if len(begin_atom_neighbors) in [2, 3]:
                acc = []
                for neighbor in begin_atom_neighbors:
                    neighbor_is_single_valenced = neighbor.GetAtomicNum() in single_valence_atoms
                    if neighbor_is_single_valenced:
                        acc.append(neighbor_is_single_valenced)

                total_num_single_valenced_neighb = len(acc)

                if ((total_num_single_valenced_neighb == 1 and begin_atom_number == 8) or
                        (total_num_single_valenced_neighb == 2 and begin_atom_number == 7)):
                    continue

        # check end atom to be methyl or substituted methyl
        if end_atom_number == 6:
            if len(end_atom_neighbors) == 4:
                acc = []
                for neighbor in end_atom_neighbors:
                    neighbor_is_single_valenced = neighbor.GetAtomicNum() in single_valence_atoms
                    if neighbor_is_single_valenced:
                        acc.append(neighbor_is_single_valenced)

                total_num_single_valenced_neighb = len(acc)

                if total_num_single_valenced_neighb >= 3:
                    continue

        # check end atom to be OH or NH2
        if end_atom_number in [8, 7]:
            if len(end_atom_neighbors)  in [2, 3]:
                acc = []
                for neighbor in end_atom_neighbors:
                    neighbor_is_single_valenced = neighbor.GetAtomicNum() in single_valence_atoms
                    if neighbor_is_single_valenced:
                        acc.append(neighbor_is_single_valenced)

                total_num_single_valenced_neighb = len(acc)

                if ((total_num_single_valenced_neighb == 1 and end_atom_number == 8) or
                        (total_num_single_valenced_neighb == 2 and end_atom_number == 7)):
                    continue

        filtered_bonds.append(bond)

    return filtered_bonds


def get_all_torsion_atoms_idx(mol: Chem.Mol, verbose: bool = False) -> list[tuple[int]]:
    """The function that generates a list with tuples of atomic ids, that specify all torsions
    in a molecule excluding such torsions as  terminal methyls, primary amines, alcohols,
    and halogens, etc.

    Parameters
    ----------
    mol : Chem.Mol
        The rdkit.Chem.Mol object
    verbose : bool, optional
        Be verbose if needed, by default False

    Returns
    -------
    list[tuple[int]]
        list with tuples of atomic ids, that specify all torsions in a molecule excluding such
    torsions as  terminal methyls, primary amines, alcohols, and halogens, etc.

    Raises
    ------
    ValueError
        If molecule is not specified
    """

    if mol is None:
        raise ValueError("Did not find any Molecule")

    rotatable_bonds = get_rotatable_bonds(mol)
    filtered_rotatable_bonds = filter_rotatable_bonds(rotatable_bonds=rotatable_bonds)

    if verbose:
        print("Found so many rotatable bonds:  ", len(filtered_rotatable_bonds))

    # Generate torsion profiles
    torsions = []
    for bond in filtered_rotatable_bonds:
        torsion_atom_ids = get_torsion_atoms_idx(bond)
        torsions.append(torsion_atom_ids)
    torsions = tuple(torsions)

    if verbose:
        print("Found torsion atoms", torsions)

    return torsions
