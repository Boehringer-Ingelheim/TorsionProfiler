# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
This is an implementation of a streamlit server
"""

import os
import ast
import tempfile
from datetime import datetime

import pandas as pd
import py3Dmol
from matplotlib import colors
import plotly.graph_objects as go

from rdkit import Chem

import streamlit as st
import streamlit.components.v1 as components

from torsion_profiler import conf
from torsion_profiler.utils import bash
from torsion_profiler.utils.metrics import get_simple_energy_classifications


@st.cache_data
def convert_df(df):
    """
    Convert the df
    """
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


@st.cache_data
def build_input_mol_view(
    mol: Chem.Mol,
    torsion_atoms: tuple[int, int, int, int],
    width: int = 600,
    height: int = 600,
)->py3Dmol.view:
    """
    Build a view for the input molecule visualization.

    Parameters
    ----------
    mol: Chem.Mol
        molecule to be visualized
    torsion_atoms: tuple[int, int, int, int]
        will be marked
    width: int
        width of view
    height: int
        height of view

    Returns
    -------
    view.html
        the html view of the py3DMol visualization
    """
    mol_block = Chem.MolToMolBlock(mol)

    view: py3Dmol.view = py3Dmol.view(width=width, height=height)
    view.addModel(mol_block, "sdf")
    view.setStyle({"model": -1}, {"stick": {}})
    view.setStyle(
        {"serial": list(torsion_atoms)},
        {"stick": {}, "sphere": {"radius": 0.62, "color": "purple", "opacity": 0.8}},
    )
    view.addPropertyLabels(
        "index",
        {"model": -1},
        {
            "fontColor": "black",
            "font": "sans-serif",
            "fontSize": 12,
            "showBackground": False,
            "alignment": "center",
        },
    )
    view.zoomTo()
    return view


# @st.cache_data
def build_torsion_profile_mols_view(
    mols: Chem.Mol,
    torsion_atoms: tuple[int, int, int, int],
    df: pd.DataFrame,
    index: int = 0,
    widht: int = 300,
    height: int = 300,
) -> py3Dmol.view:
    """
    Build result view of torsion profile calculatoin.

    Parameters
    ----------
    mol: Chem.Mol
        molecule to be visualized
    torsion_atoms: tuple[int, int, int, int]
        will be marked
    df: pd.DataFrame
        results of tors profile.
    index:int
        index of conformer to view
    width: int
        width of view
    height: int
        height of view


    Returns
    -------
    py3Dmol.view
        the view of the py3DMol visualization
    """

    models = ""
    for mol in mols:
        mol_block = Chem.MolToMolBlock(mol)
        models += str(mol_block)
        models += "$$$$\n"

    energies = df["rel_potential_energy"]

    vcolors = get_simple_energy_classifications(energies)

    view = py3Dmol.view(width=widht, height=height)
    view.addModelsAsFrames(models, "sdf")
    view.setStyle({"model": -1}, {"stick": {}})

    if "animate" not in st.session_state:
        st.session_state["animate"] = False

    if "struct_frame" not in st.session_state:
        st.session_state["struct_frame"] = None

    if st.session_state["animate"]:
        view.setStyle(
            {"model": -1, "serial": list(torsion_atoms)},
            {"stick": {}, "sphere": {"radius": 0.6, "color": "purple", "opacity": 0.8}},
        )
        view.animate({"loop": "forward", "interval": 200, "step": 1})
        view.pauseAnimate()
    else:
        view.setStyle(
            {"model": -1, "serial": list(torsion_atoms)},
            {
                "stick": {},
                "sphere": {"radius": 0.6, "color": vcolors[index], "opacity": 0.8},
            },
        )

    if st.session_state["struct_frame"] is not None:
        view.setFrame(st.session_state["struct_frame"])

    return view


def main():
    """this is the main routine for the server."""
    root_dir = os.path.dirname(__file__)
    st.set_page_config(
        page_title="torsion_profiler - TorsionProfiler - Prototype",
        page_icon=":microscope:",
        layout="wide",
        menu_items={
        "Report A Bug": "https://github.com/Boehringer-Ingelheim/TorsionProfiler/issues",
        "About": "https://github.com/Boehringer-Ingelheim/TorsionProfiler",
        },
    )

    h_col1, h_col2 = st.columns(2)
    h_col1.title("TorsionProfiler")
    h_col2.image(
        root_dir + "/../../.img/molecule_flowers_small_single.png",
        width=80,
    )

    if "df" in st.session_state:
        tab1 = st.expander("PREPARE", expanded=False)
        tab2 = st.expander("RESULTS", expanded=True)
    else:
        tab2 = st.expander("RESULTS", expanded=False)
        tab1 = st.expander("PREPARE", expanded=True)

    with tab1:
        # tab1.header("Prepare")
        tab1.markdown("Input")
        col1, col2 = st.columns(2)

        torsion_atoms = ast.literal_eval(
            col1.text_input("Torsion Atom IDs", value="0,1,2,3")
        )
        mol_smi = col2.text_input("Mol Smiles", value="CCCC")

        sdf_file = col2.file_uploader(
            "Or upload .sdf",
        )

        if sdf_file is not None and not isinstance(sdf_file, int):
            with tempfile.NamedTemporaryFile(
                "wb", dir=os.getcwd(), suffix=".sdf", prefix="torsion_profiler_tmp_"
            ) as tmp:
                # Write out
                lines = sdf_file.readlines()
                tmp.writelines(lines)
                tmp.flush()

                # Load mols
                gen = Chem.SDMolSupplier(tmp.name, removeHs=False)
                mol = next(gen)

        else:
            mol = Chem.MolFromSmiles(mol_smi)
            mol = Chem.AddHs(mol)
        Chem.rdDistGeom.EmbedMolecule(mol)

        container_width = 250
        view1 = build_input_mol_view(
            mol,
            torsion_atoms=torsion_atoms,
            width=container_width,
            height=container_width,
        )
        with col1:
            html_view1 = view1._make_html()
            components.html(html_view1, width=container_width, height=container_width)

        st.markdown("Settings")
        settings_col1, settings_col2, _ = st.columns(3)
        method = settings_col1.radio(
            "Calculator?", ["OFF", "ANI2X", "XTB"], index=0, horizontal=True
        )
        optimize = ast.literal_eval(
            settings_col1.radio("Optimize?", ["False", "True"], index=1, horizontal=True)
        )
        n_measurements = settings_col2.number_input(
            "n Measurements", min_value=0, max_value=360, value=13
        )

        # Calculate
        # Setup machinery
        def do_the_math(mol: Chem.Mol, torsion_atoms: tuple[int, int, int, int]):
            """
            does some math - calculate torsoin profile.

            Parameters
            ----------
            mol: Chem.Mol
                input mol
            torsion_atoms: tuple[int, int, int, int]
                torsion defining atom ids

            Returns
            -------
            None
            """
            from torsion_profiler.tools.torsion_profiler import TorsionProfiler
            from torsion_profiler.orchestration.submission_systems import Local

            env = os.environ
            if method == "ANI2X":
                env.update({"OPENMM_CPU_THREADS": "4"})
                from torsion_profiler.engines import AniCalculator

                conda_env_path = conf["conda_calculator_envs"][AniCalculator.__name__]
                submission_system = Local(
                    submission=True,
                    conda_env_path=conda_env_path,
                    environment=env,
                )
                out_file_prefix = "tmp_torsion_profilerStreamlit_ani"
                out_folder_path = os.getcwd()
                calculator = AniCalculator(optimize_structure=optimize)

            elif method == "OFF":
                env.update({"OPENMM_CPU_THREADS": "4"})
                from torsion_profiler.engines import OffCalculator

                conda_env_path = conf["conda_calculator_envs"][OffCalculator.__name__]
                submission_system = Local(
                    submission=True,
                    conda_env_path=conda_env_path,
                    environment=env,
                )
                out_file_prefix = "tmp_torsion_profilerStreamlit_off"
                out_folder_path = os.getcwd()
                calculator = OffCalculator(optimize_structure=optimize)

            elif method == "XTB":
                from torsion_profiler.engines.xtb_calculator import XtbCalculator
                conda_env_path = conf["conda_calculator_envs"][XtbCalculator.__name__]

                submission_system = Local(
                    submission=True,
                    conda_env_path=conda_env_path,
                )
                out_file_prefix = "tmp_torsion_profilerStreamlit_xtb"
                out_folder_path = os.getcwd()
                calculator = XtbCalculator(optimize_structure=optimize)

            tp = TorsionProfiler(
                calculator=calculator,
                submission_system=submission_system,
                n_measurements=n_measurements,
                _force=True
            )

            st.session_state["animate"] = False

            tp.verbose = True
            # spin = st.spinner("calculating: ")
            start = datetime.now()
            df = None
            with tab1:
                with st.spinner("calculating: " + str(method)):
                    progress_bar = st.progress(0)
                    df = tp.calculate_torsion_profile(
                        mol=mol, torsion_atom_ids=torsion_atoms,
                        out_dir=out_folder_path, approach_name=out_file_prefix
                    )
                    progress_bar.progress(100)

            if submission_system is not None:
                bash.rmtree(out_folder_path + "/" + out_file_prefix)

            fun = st.balloons()
            end = datetime.now()
            duration = end - start
            st.session_state["df"] = df[[c for c in df.columns if c not in ["ROMol"]]]
            st.session_state["mol_prof"] = tp.out_tp["ROMol"]
            st.session_state["torsion_atoms"] = torsion_atoms
            st.session_state["runTime"] = (
                str(duration.seconds // 60) + "min " + str(duration.seconds % 60) + "s"
            )

            del fun

        st.button(
            "Calculate",
            on_click=do_the_math,
            args=(mol, torsion_atoms),
        )

    if "df" in st.session_state:  # and st.session_state['df'] is not None
        # Results
        with tab2:
            tab1.expanded = False
            tab2.header("Result")
            tab2_col1, tab2_col2 = tab2.columns(2)
            tab2.markdown("run time: " + st.session_state["runTime"])
            with tab2_col1:
                width = 300
                height = 300
                index = st.slider(
                    "Conformer id",
                    0,
                    len(st.session_state["mol_prof"]),
                    1,
                    disabled=st.session_state["animate"],
                )

                def set_animation():
                    if "animate" not in st.session_state:
                        st.session_state["animate"] = False

                    if st.session_state["animate"]:
                        st.session_state["animate"] = False
                        st.session_state["struct_frame"] = index
                    else:
                        st.session_state["animate"] = True
                        st.session_state["struct_frame"] = index

                st.button("play/pause", on_click=set_animation)

                st.session_state["struct_frame"] = index
                view = build_torsion_profile_mols_view(
                    st.session_state["mol_prof"],
                    torsion_atoms=st.session_state["torsion_atoms"],
                    df=st.session_state["df"],
                    index=st.session_state["struct_frame"],
                )
                html_js = view._make_html()
                components.html(html_js, width=width, height=height)

            with tab2_col2:
                angles = st.session_state["df"]["torsion_angle"].to_numpy()
                v = st.session_state["df"]["rel_potential_energy"].clip(0,30).to_numpy()

                fig = go.Figure(
                    data=go.Scatter(
                        x=angles,
                        y=v,
                        mode="lines+markers",
                        marker={"size":10, "color":get_simple_energy_classifications(v)},
                        line={"color":"grey"},
                        name="torsion profile",
                        hoverinfo="y",
                    )
                )

                fig.update_layout(
                    margin={"l":0, "r":0, "t":20, "b":0},
                )
                if not st.session_state["animate"]:
                    fig.add_trace(
                        go.Scatter(
                            x=[angles[index]],
                            y=[v[index]],
                            mode="markers",
                            marker={
                                "size":20,
                                "color":"rgba(0, 0, 0,0.0)",
                                "line":{"color": "rgb(255, 165, 0.8)", "width": 6},
                            },
                            name="selected",
                        )
                    )

                tab2_col2.plotly_chart(fig)

            csv = convert_df(st.session_state["df"])
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="torsion_profile.tsv",
                mime="text/csv",
            )
            if not st.session_state["animate"]:
                selected_enery = get_simple_energy_classifications(
                    [st.session_state["df"]["rel_potential_energy"][index]]
                )
                color_d = {
                    "background-color": "rgb"
                    + str(
                        tuple(
                            map(
                                lambda x: x * 256,
                                colors.to_rgb(
                                    selected_enery[0],
                                ),
                            )
                        )
                    ).replace(")", ", 0.4)")
                }
                series = (
                    st.session_state["df"]
                    .style.format(
                        subset=["torsion_angle", "potential_energy", "rel_potential_energy"],
                        formatter="{:.2f}",
                    )
                    .set_properties(subset=pd.IndexSlice[[index], :], **color_d)
                )
            else:
                series = st.session_state["df"].style.format(
                    subset=["torsion_angle", "potential_energy", "rel_potential_energy"],
                    formatter="{:.2f}",
                )
            tab2.dataframe(
                series,
                use_container_width=True,
            )
            print(series)


if __name__ == "__main__":
    main()
