# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
Parse Molecule input
"""
from rdkit import Chem
from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED


def _load_molecule_from_smiles(user_input, context):
    """
    Loader for a smiles input
    """
    # MolFromSmiles returns None if the string is not a molecule
    # after either redirect_stdout or redirect_stderr.
    mol = Chem.MolFromSmiles(user_input)
    if mol is None:
        return NOT_PARSED

    Chem.rdDepictor.Compute2DCoords(mol)
    return mol


def _load_molecule_from_sdf(user_input, context):
    """
    Loader for a sdf file input
    """
    if ".sdf" not in str(user_input):  # this silences some stderr spam
        return NOT_PARSED

    try:
        mols = list(Chem.SDMolSupplier(user_input, removeHs=False))
        return mols
    except ValueError:  # any exception should try other strategies
        return NOT_PARSED


def _load_molecule_from_chem(user_input, context):
    """
    Loader for a rdkit mol
    """
    if isinstance(user_input, Chem.Mol):  # this silences some stderr spam
        return [user_input]

    if isinstance(user_input, list) and all(isinstance(m, Chem.Mol) for m in user_input):
        return user_input

    return NOT_PARSED


def _load_molecule_from_mol2(user_input, context):
    """
    Loader for a mol2 input file
    """
    if ".mol2" not in str(user_input):
        return NOT_PARSED

    m = Chem.MolFromMol2File(user_input)
    if m is None:
        return NOT_PARSED
    return m


get_molecule = MultiStrategyGetter(
    strategies=[
        _load_molecule_from_sdf,
        _load_molecule_from_mol2,
        _load_molecule_from_chem,
        _load_molecule_from_smiles,
    ],
    error_message="Unable to generate a molecule from '{user_input}'.",
)

MOL = Option(
    "-m",
    "--mol",
    help=("SmallMoleculeComponent. Can be provided as an SDF file or as a SMILES " " string."),
    getter=get_molecule,
)
