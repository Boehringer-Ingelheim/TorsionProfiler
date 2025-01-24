"""
This module defines a jnb widget for showing multiple torsion profiles
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from ipywidgets import widgets
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from .visualize_torsion_profiles import _show_mol_2d


def _draw_profiles_helper(master_df: pd.DataFrame, i: int, methods:list[str], c_dict:dict[str, ]):
    """
    this helper function is esd as updater.

    Parameters
    ----------
    master_df: pd.DataFrame
        contains all torsion profiles
    i: int
        index of the torsion profile
    methods
        selected method
    c_dict: dict[str,]
        color definition for methods

    Returns
    -------
    plt.Figure
        returns a matplotlib figure.

    """
    fig = plt.figure(figsize=[12, 9])
    ax = fig.gca()

    mol_id = master_df.molID.unique()[i]
    mol_data = master_df.where(master_df.molID == mol_id).dropna()

    ax2 = plt.axes([0.9, 0.1, 0.38, 0.38], frameon=False)
    mol = Chem.MolFromSmiles(mol_data.molecule.iloc[0])
    torsion_atom_ids = mol_data.torsion_atom_ids.iloc[0]
    _show_mol_2d(mol=mol, torsion_atom_ids=torsion_atom_ids, ax=ax2)

    for method in mol_data.approach.unique():
        if method not in methods:
            continue

        data = mol_data.where(mol_data.approach == method).dropna()

        x = data.angles
        y = data.rel_potential_energies.astype(float)
        x, y = np.array(list(sorted(zip(x, y), key=lambda t: t[0]))).T
        ax.plot(
            x,
            y,
            label=method,
            c=c_dict[method],
            marker="o",
        )

    ax.legend(bbox_to_anchor=(1.425, 1), loc="upper right", ncol=1, fontsize=20)

    if np.max(mol_data.rel_potential_energies) < 10:
        ax.set_ylim([0, 10])

    ymax = 10
    ax.axhspan(0, 2.5, facecolor="green", alpha=0.15)
    ax.axhspan(2.5, 5, facecolor="gold", alpha=0.15)
    ax.axhspan(5, 8, facecolor="orange", alpha=0.25)
    if ymax * 1.3 > 8:
        ax.axhspan(8, ymax * 1.3, facecolor="red", alpha=0.3)

    ax.set_ylabel("$V~[kcal/mol]$", fontsize=32)
    ax.set_xlabel("$angles~[deg]$", fontsize=32)
    ax.set_title(mol_data.molID.iloc[0], fontsize=36)

    return fig


def display_profiles(
    master_df: pd.DataFrame, start_val: int = 0, colors=None
) -> widgets.VBox:
    """
    Implements the IPython Widget to visualize multiple torsion profiles.

    Parameters
    ----------
    master_df: pd.DataFrame
        dataframe containing multiple torsion profiles
    start_val: int
        start index for the torsion profiles.
    colors
        iterables

    Returns
    -------
    widgets.VBox
        the torsion profile widget
    """
    if colors is None:
        colors = mcolors.TABLEAU_COLORS
    methods = list(master_df.approach.unique())
    c_dict = dict(zip(methods, colors))

    def display_edge(index, method):
        fig = _draw_profiles_helper(master_df, index, method, c_dict)
        return fig

    slider = widgets.IntSlider(
        tooltip="select molecule",
        description="mol index",
        min=0,
        max=len(master_df.molID.unique()) - 1,
        step=1,
        value=start_val,
    )

    next_button = widgets.Button(tooltip="next structure", icon="caret-right")
    def increment():
        if slider.value == slider.max:
            slider.value = 0
        else:
            slider.value += 1

    next_button.on_click(increment)

    previous_button = widgets.Button(tooltip="previous structure", icon="caret-left")
    def decrement():
        if slider.value == 0:
            slider.value = slider.max
        else:
            slider.value -= 1

    previous_button.on_click(decrement)

    all_methods = master_df.approach.unique()
    unique_mol = list(master_df.molID.unique())[slider.value]
    approaches = master_df.where(master_df.molID == unique_mol).dropna().approach.unique()
    print(approaches)

    w = widgets.SelectMultiple(
        options=list(all_methods),
        value=list(approaches),
        rows=len(approaches),
        description="Methods",
        disabled=False,
    )

    hbox = widgets.HBox([previous_button, next_button, slider])
    inter = widgets.interactive_output(display_edge, {"index": slider, "method": w})

    vbox = widgets.VBox([hbox, widgets.HBox([inter, w])])
    return vbox
