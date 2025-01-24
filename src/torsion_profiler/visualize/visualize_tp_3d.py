"""
This file implements 3D visualizations
"""

import tempfile
from time import sleep
from typing import Union, Iterable

from rdkit import Chem

import nglview as ng
import mdtraj

from ..utils import read_mol_db


def show_3d_mol(
    mol: Union[str, Chem.Mol],
    highlight_atoms: list[int] = None,
    label_atom_ids: bool = True,
    color: Union[str, Iterable[str]] = "green",
    overlay_states: bool = False,
) -> ng.widget.NGLWidget:
    """
    3D representation of the input molecule. can be easily visualized in jupyter notebooks.
    color a target torsion with atom_torsion ids in a desired color (can be multiple)

    Parameters
    ----------
    mol : Union[str, Chem.Mol]
        molecule containing multiple conformers. can be Chem.Mol or str path to a sdf or pdb file.
    highlight_atoms : tuple[int, int, int, int], optional
        tuple containing the atom ids to be highlighted, by default None
    label_atom_ids : bool, optional
        label atoms with their id, by default True
    color : Union[str, Iterable[str]], optional
        color highlighting for the torsion atoms. (if multiple, frames will be colored differently),
         by default "green"

    Returns
    -------
    ng.widget.NGLWidget
        3D view of the molecule

    Raises
    ------
    ValueError
        if colors were not understood
    """
    in_pdb=None
    # build tmp pdb
    if isinstance(mol, Iterable):
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix="_torsion_profiler_vis.pdb")
        pdb_blocks = [ Chem.MolToPDBBlock(m) for m in mol]
        tmp_file.write("\n".join(pdb_blocks))
        in_pdb= tmp_file.name
    if isinstance(mol, Chem.Mol):
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix="_torsion_profiler_vis.pdb")
        in_pdb = tmp_file.name
        Chem.MolToPDBFile(mol, in_pdb)
    elif isinstance(mol, str) and mol.endswith(".sdf"):
        mol = read_mol_db(mol)
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix="_torsion_profiler_vis.pdb")
        in_pdb = tmp_file.name
        Chem.MolToPDBFile(mol, in_pdb)
    elif isinstance(mol, str) and mol.endswith(".pdb"):
        in_pdb = mol
        tmp_file = None

    traj = mdtraj.load_pdb(in_pdb)

    if tmp_file is not None:
        tmp_file.close()

    # Generate view
    if overlay_states:
        view = ng.show_mdtraj(traj[0])
        components = [view]
        for frame in traj[1:]:
            c = view.add_component(frame)
            components.append(c)
    else:
        view = ng.show_mdtraj(traj)

    sleep(1)
    view.clear()
    sleep(1)

    # manage representation
    view.clear_representations()
    sleep(1)

    # Color Torsion
    if highlight_atoms is not None:
        view.add_representation("hyperball")
        if isinstance(color, str):  # if single color
            view.add_ball_and_stick(
                color=color,
                selection="@" + ",".join(map(str, highlight_atoms)),
                opacity=0.15,
                radius="0.35",
            )
        elif isinstance(color, list) and len(color) == len(traj):  # if multi colored
            view.add_ball_and_stick(
                color=color[0],
                selection="@" + ",".join(map(str, highlight_atoms)),
                opacity=0.35,
                radius="0.35",
            )
            view.color = color

            # Event listener
            def on_change(change):
                frame = change.new
                fcolor = view.color[int(frame)]
                view.update_ball_and_stick(color=fcolor)
                sleep(0.01)  # wait for the color update

            view.observe(on_change, names=["frame"])
        else:
            raise ValueError(
                "Stupido! Maybe color not correct type or multi colors does not match len(traj)"
            )
    elif color is not None and len(color) == len(traj):
        for i, c in enumerate(color):
            components[i].add_representation("hyperball", color=c)
    else:
        view.add_representation("hyperball")

    # Atom labels
    if label_atom_ids:
        view.add_label(
            color="black",
            scale=0.75,
            labelType="text",
            labelText=[str(a.index) for a in traj.topology.atoms],
            attachment="middle_center",
        )

    return view


def show_3d_torsion_flower(in_mol: Chem.Mol) -> ng.widget.NGLWidget:
    """
    draw all conformers of a rdkit molecule with this as an NGL widget

    Parameters
    ----------
    in_mol : Chem.Mol
        molecule containing multiple conformers

    Returns
    -------
    ng.widget.NGLWidget
        the beautiful torison flower
    """
    if isinstance(in_mol, Chem.Mol):
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix="_torsion_profiler_vis.pdb")
        in_pdb = tmp_file.name
        Chem.MolToPDBFile(in_mol, in_pdb)
    if isinstance(in_mol, str) and in_mol.endswith(".sdf"):
        in_mol = read_mol_db(in_mol)
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix="_torsion_profiler_vis.pdb")
        in_pdb = tmp_file.name
        Chem.MolToPDBFile(in_mol, in_pdb)
    elif isinstance(in_mol, str) and in_mol.endswith(".pdb"):
        in_pdb = in_mol
        tmp_file = None

    # generate View
    traj = mdtraj.load_pdb(in_pdb)
    if tmp_file is not None:
        tmp_file.close()

    view = ng.show_mdtraj(traj[0])
    sleep(1)

    # manage representation
    view._remove_representation()
    view.add_hyperball()

    # Add all conformers
    for i, _ in enumerate(traj):
        sleep(0.2)
        view.add_component(traj[i])
        view._remove_representation(component=i + 1)
        view.add_hyperball(component=i + 1)

    return view
