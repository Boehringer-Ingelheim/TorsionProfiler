"""
This module is used to plot torsion profiles
"""

import ast
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib import pyplot as plt

# 2D
from typing import Iterable, Union


def plot_torsion_profile(
    tp_df: pd.DataFrame,
    out_fig_path: str = None,
    mol_id: str = "",
    title_prefix: str = "",
    color: Iterable[str] = None,
    ylim: tuple[int, int] = None,
) -> plt.Figure:
    """
    with this function thetorsion profile can be plotted.

    Parameters
    ----------
    tp_df: pd.DataFrame
    mol: Chem.Mol
    out_fig_path: str
    mol_id: str
    title_prefix: str
    color: Iterable[str]
    ylim: tuple[int, int]

    Returns
    -------
    plt.Figure
        figure of the plot
    """
    columns = tp_df.columns

    mol = tp_df["ROMol"].iloc[0]
    torsion_atom_ids = ast.literal_eval(str(tp_df["torsion_atom_ids"].iloc[0]))

    if "torsion_angle2" in columns:
        if len(torsion_atom_ids) > 0:
            torsion_atom_ids2 = ast.literal_eval(str(tp_df["torsion_atom_ids_2"].iloc[0]))
        else:
            torsion_atom_ids2 = []
        df = tp_df.sort_values(ascending=True, by=["torsion_angle2", "torsion_angle"])
        x_angles = df.angles.unique().round()
        y_angles = df.angles2.unique().round()
        x_id_angles = {a: i for i, a in enumerate(x_angles) if not np.isnan(a)}
        y_id_angles = {a: i for i, a in enumerate(y_angles) if not np.isnan(a)}

        nx_angle = len(x_id_angles)  # PROBLEM!
        ny_angle = len(y_id_angles)

        v_landscape = np.zeros((nx_angle, ny_angle))

        data = df[["torsion_angle", "torsion_angle2", "rel_potential_energy"]]
        for i, row in data.iterrows():
            if not np.isnan(row.angles):
                v_landscape[x_id_angles[row["torsion_angle"]], y_id_angles[row[
                    "torsion_angle2"]]] = (
                    row.rel_potential_energies
                )

        fig = _plot_2d_torsion_profile_with_mol(
            mol=mol,
            torsion_atom_ids1=torsion_atom_ids,
            torsion_atom_ids2=torsion_atom_ids2,
            v_landscape=v_landscape,
            x_angles=x_angles,
            y_angles=y_angles,
            title_prefix=title_prefix,
            out_fig_path=out_fig_path,
            mol_id=mol_id,
        )
    else:
        if "approach" in columns:
            x = []
            y = []
            data_labels = []
            for label in tp_df["approach"].unique():
                data_labels.append(label)
                app_data = tp_df.loc[tp_df["approach"] == label]
                app_data = app_data.sort_values(by="torsio  n_angle")
                x.append(app_data["torsion_angle"].astype(float).to_numpy())
                y.append(app_data["rel_potential_energy"].astype(float).to_numpy())

        else:
            tp_df = tp_df.sort_values(by="torsion_angle")
            x = tp_df["torsion_angle"].astype(float).to_numpy()
            y = tp_df["rel_potential_energy"].astype(float).to_numpy()
            data_labels = []
            print(x,y)

        fig = _plot_torsion_profile_with_mol(
            mol=mol,
            torsion_atom_ids=torsion_atom_ids,
            x=x,
            y=y,
            title_prefix=title_prefix,
            data_labels=data_labels,
            color=color,
            out_fig_path=out_fig_path,
            mol_id=mol_id,
            ylim=ylim,
        )

    return fig


def _plot_torsion_profile_with_mol(
    mol: Chem.Mol,
    torsion_atom_ids: tuple[int, int, int, int],
    x: np.array,
    y: np.array,
    title_prefix: str = "",
    data_labels: Iterable[str] = "",
    color: Iterable[str] = None,
    out_fig_path: str = None,
    mol_id: str = "",
    ylim: tuple[int, int] = None,
) -> plt.figure:
    """
    This function plots the molecule with an annotated torsion and its torsion rotation profile.

    Parameters
    ----------
    mol : Chem.Mol
        molecule, that is represented in x and y
    torsion_atom_ids : tuple[int, int, int, int]
        tuple giving the atom ids
    x : np.array
        angles can be an np.array[float] for a single data set or np.array[np.array[float]]
        for multiple datasets, to produce an overlay
    y : np.array
        energies can be an np.array[float] for a single data set or np.array[np.array[float]]
        for multiple datasets, to produce an overlay
    title_prefix : str, optional
        titel of the plot (e.g. torsion atom ids), by default ""
    data_labels : Iterable[str], optional
        labels for the different datasets, by default ""
    color : Iterable[str], optional
        optional colors for points, by default None
    out_fig_path: str, optional
        write out the resulting figure to this path, by default None
    mol_id: str, optional
        Specify if needed the ID of the molecule of interest

    Returns
    -------
    plt.figure
        returns a matplotlib figure
    """

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=[8, 5], gridspec_kw={"width_ratios": [1, 3]})
    fig.set_facecolor("white")
    axes = np.squeeze(np.array(axes, ndmin=2))

    # plot mol
    _show_mol_2d(mol=mol, torsion_atom_ids=torsion_atom_ids, ax=axes[0])

    # plot profiles
    _plot_torsion_profile(
        x,
        y,
        labels=data_labels,
        title_prefix=title_prefix,
        ax=axes[1],
        color=color,
        mol_id=mol_id,
        ylim=ylim,
    )

    fig.tight_layout()

    if out_fig_path is not None:
        fig.savefig(out_fig_path, dpi=400)

    return fig


def _plot_2d_torsion_profile_with_mol(
    mol: Chem.Mol,
    torsion_atom_ids1: tuple[int, int, int, int],
    torsion_atom_ids2: tuple[int, int, int, int],
    v_landscape: np.array,
    x_angles=None,
    y_angles=None,
    title_prefix: str = "",
    out_fig_path: str = None,
    mol_id: str = "",
):
    """
    This function plots the molecule with an annotated torsion and its torsion rotation profile.

    Parameters
    ----------
    mol : Chem.Mol
        molecule, that is represented in x and y
    torsion_atom_ids : tuple[int, int, int, int]
        tuple giving the atom ids
    v_landscape : np.array
        angles can be an np.array[float] for a single data set or np.array[np.array[float]]
        for multiple datasets, to produce an overlay
    title_prefix : str, optional
        titel of the plot (e.g. torsion atom ids), by default ""
    out_fig_path: str, optional
        write out the resulting figure to this path, by default None
    mol_id: str, optional
        Specify if needed the ID of the molecule of interest

    Returns
    -------
    plt.figure
        returns a matplotlib figure
    """
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=[8, 5], gridspec_kw={"width_ratios": [1, 3]})
    fig.set_facecolor("white")
    axes = np.squeeze(np.array(axes, ndmin=2))

    # plot mol
    _show_mol_2d(
        mol=mol, torsion_atom_ids=list(torsion_atom_ids1) + list(torsion_atom_ids2), ax=axes[0]
    )

    # plot profiles
    if x_angles is None:
        x_angles = np.arange(v_landscape.shape[0])
    if y_angles is None:
        y_angles = np.arange(v_landscape.shape[1])

    _plot_2dtorsion_profile(
        v_landscape,
        x_angles=x_angles,
        y_angles=y_angles,
        title_prefix=title_prefix,
        ax=axes[1],
        mol_id=mol_id,
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.85)

    if out_fig_path is not None:
        fig.savefig(out_fig_path, dpi=400)

    return fig


def _plot_torsion_profile(
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str] = None,
    title_prefix: str = "",
    out_fig_path: str = None,
    color: str = None,
    color_scale_background: bool = True,
    ax: plt.Axes = None,
    mol_id: str = "",
    ylim: tuple[int, int] = None,
) -> plt.figure:
    """This function is used to plot one or multiple scans of the same dihedral angle

    Parameters
    ----------
    x : np.ndarray
        2D array. Elements of this array are arrays of scanned angles
    y : np.ndarray
        2D array. Elements of this array are arrays of the corresponding energies
    labels : list[str], optional
        Labels for every scan, by default []
    title_prefix : str, optional
        Specify if needed which atoms defined the scanned dihedral angle, by default ""
    out_fig_path : str, optional
        If you want to save the resulting plot - set the saving path including the figure's name
        and its extension, by default None. By default the figure is not saved
    mol_id : str, optional
        Specify if needed the ID of molecule of interest

    Returns
    -------
    plt.figure
         Returns the plot of the scanned angles against corresponding energies
    """
    if (hasattr(x, "shape") and len(x.shape) == 1) or (isinstance(x, np.ndarray) and x.ndim == 1):
        x = np.array(x, ndmin=2)
    if (hasattr(y, "shape") and len(y.shape) == 1) or (isinstance(y, np.ndarray) and y.ndim == 1):
        y = np.array(y, ndmin=2)

    if isinstance(labels, str):
        labels = [labels]
    if isinstance(labels, str) and len(labels) != len(x):
        labels = ["set_" + str(i) for i in range(1, len(x) + 1)]
    elif(labels is None):
        labels=[]
    else:
        labels = list(map(str, labels))

    if isinstance(color, str):
        color = [color for x in range(len(x))]
    elif isinstance(color, list) and all(isinstance(x, str) for x in color):
        color = [color for x in range(len(x))]
    elif color is None:
        color = [color for x in range(len(x))]

    ymax = 0
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = None

    for i, (x_set, y_set, c) in enumerate(zip(x, y, color)):
        if len(y_set) == 0:
            continue
        ymax = max(ymax, np.max(y_set))

        if len(labels) > i:
            lab = labels[i]
            ax.plot(x_set, y_set, marker="o", markersize=3, label=lab)
        else:
            ax.plot(x_set, y_set, marker="o", markersize=3,)
        xticks = np.linspace(-180, 180, 7)
        ax.set_xticks(ticks=xticks)
        ax.set_xlim([-180, 180])

        if c is not None:
            ax.scatter(x_set, y_set, marker="o", s=120, c=c, alpha=1)

    ymax = max(ymax, 4)
    if color_scale_background:
        ax.axhspan(0, 2.5, facecolor="green", alpha=0.1, zorder=-100)
        ax.axhspan(2.5, 5, facecolor="gold", alpha=0.1, zorder=-100)
        ax.axhspan(5, 8, facecolor="orange", alpha=0.2, zorder=-100)
        cymax = max(ymax, 40)
        ax.axhspan(8, cymax * 1.3, facecolor="red", alpha=0.2, zorder=-100)

    if ylim is None:
        ax.set_ylim([0, ymax * 1.15])
    else:
        ax.set_ylim(ylim)

    ax.set_title(str(title_prefix) + " Dihedral Angle " + str(mol_id))
    ax.set_xlabel("Angle [$\degree$]")
    ax.set_ylabel("Relative energy [kcal/mol]")
    if len(labels) > 0:
        ax.legend()

    if fig is not None:
        fig.tight_layout()

    if out_fig_path is not None:
        fig.savefig(out_fig_path)

    return fig


def _show_mol_2d(
    mol: Chem.Mol,
    torsion_atom_ids: tuple[int, int, int, int],
    out_fig_path: str = None,
    ax: plt.Axes = None,
) -> Union[plt.figure, None]:
    """
    plot the molecule in 2D and annotate the torsion angle. If an axes is provided, the mol will
    be added to it.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule that shall be drawn in 2D
    torsion_atom_ids : tuple[int, int, int, int]
        annotate this torsion angle
    out_fig_path : str, optional
        save figure in this path, by default None
    ax : plt.Axes, optional
        plot the image to this axes (no figure is generated then), by default None

    Returns
    -------
    Union[plt.figure, None]
        figure of the plot or None if ax was provided.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = None

    # Plot mol:
    mol_copy = Chem.Mol(mol)
    mol_copy = Chem.RemoveHs(mol_copy, implicitOnly=False, updateExplicitCount=False,
                             sanitize=False)
    mol_copy.RemoveAllConformers()

    bonds = []
    color_ta=[]
    if torsion_atom_ids is not None:
        for i in range(len(torsion_atom_ids) - 1):
            try:
                bond = mol.GetBondBetweenAtoms(torsion_atom_ids[i], torsion_atom_ids[i + 1])
                if bond is not None:
                    bonds.append(bond.GetIdx())
            except ValueError:
                pass
            try:
                mol_copy.GetAtomWithIdx(i)
                color_ta.append(i)
            except Exception:
                pass
        try:
            mol_copy.GetAtomWithIdx(torsion_atom_ids[-1])
            color_ta.append(torsion_atom_ids[-1])
        except Exception:
            pass
    print(torsion_atom_ids)
    g = Draw.MolToImage(
        mol_copy,
        size=(300, 300),
        canvas=None,
        fitImage=True,
        #highlightAtoms=torsion_atom_ids,
        highlightBonds=bonds,

    )
    g = np.asarray(g)
    ax.imshow(g)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    if fig is not None:
        fig.tight_layout()

    if out_fig_path is not None:
        fig.savefig(out_fig_path)

    return fig


def _plot_2dtorsion_profile(
    x_y: np.ndarray,
    x_angles: list,
    y_angles: list,
    title_prefix: str = "",
    out_fig_path: str = None,
    vmin=0,
    vmax=10,
    ax: plt.Axes = None,
    mol_id: str = "",
) -> plt.figure:
    """This function is used to plot one or multiple scans of the same dihedral angle

    Parameters
    ----------
    x_y : np.ndarray
        2D array. Elements of this array are arrays of scanned angles. Always Assume same size
        for x,y
    z : np.ndarray
        2D array. Elements of this array are arrays of the corresponding energies
    labels : list[str], optional
        Labels for every scan, by default []
    title_prefix : str, optional
        Specify if needed which atoms defined the scanned dihedral angle, by default ""
    out_fig_path : str, optional
        If you want to save the resulting plot - set the saving path including the figure's name
        and its extension, by default None. By default the figure is not saved
    mol_id : str, optional
        Specify if needed the ID of molecule of interest

    Returns
    -------
    plt.figure
         Returns the plot of the scanned angles against corresponding energies
    """
    x_y = np.array(x_y, ndmin=2)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = None

    c = ax.imshow(x_y, vmin=vmin, vmax=vmax, interpolation="nearest", origin="lower")

    n = 6
    step = len(x_angles) // n
    ax.set_yticks(range(len(y_angles))[::step], y_angles[::step])
    ax.set_xticks(range(len(x_angles))[::step], x_angles[::step], rotation=90)

    cax = ax.get_figure().add_axes(
        [ax.get_position().x1, ax.get_position().y0 + 0.025, 0.045, ax.get_position().height]
    )
    cb = plt.colorbar(c, cax=cax)

    ax.set_title(str(title_prefix) + " 2D Torsion Profile " + str(mol_id))
    ax.set_xlabel("Angle 1 [$\degree$]")
    ax.set_ylabel("Angle 2 [$\degree$]")
    cb.set_label("Relative energy [kcal/mol]")

    if fig is not None:
        fig.tight_layout()

    if out_fig_path is not None:
        fig.savefig(out_fig_path)

    return fig, cb
