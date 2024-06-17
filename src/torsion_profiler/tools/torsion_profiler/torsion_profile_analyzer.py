"""
Torsion profile analyzer
"""
import logging
from typing import Union, Callable
import multiprocessing as mult

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import integrate
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from ...utils import read_mol_db
from ...utils import metrics, bash
from ...utils.molecule_attribs import get_rotatable_bonds, filter_rotatable_bonds

log = logging.getLogger(__name__)


def thread_calculate_distances(job_id, reference_df, decoy_data, batch_i, ta_ks):
    """
    helper function for parallelization

    Parameters
    ----------
    job_id
    reference_df
    decoy_data
    batch_i
    ta_ks

    Returns
    -------

    """
    print(f"Job: {job_id} - START")
    reference_df = deepcopy(reference_df)
    decoy_data = deepcopy(decoy_data)
    ta = TorsionProfileAnalyzer()

    for k, v in ta_ks.items():
        setattr(ta, k, deepcopy(v))

    arti_ds = []
    for i in tqdm(batch_i, desc=f"Job {job_id}"):
        df = decoy_data[i]
        d = ta.build_comparison_descriptors(ref_profiles=reference_df, method_profiles=df)
        d["method_type"] = df.approach.iloc[0]
        d["ref_type"] = reference_df.approach.iloc[0]

        if "class" in df.columns:
            d["class"] = df["class"].iloc[0]
        if "set" in df.columns:
            d["set"] = df["set"].iloc[0]

        arti_ds.append(d)
    print(f"Job: {job_id} - DONE")

    return arti_ds


class TorsionProfileAnalyzer:
    """Torsion Profile Analysis tools"""
    def __init__(self, n_processes: int = 1):
        self.n_processes = n_processes
        self._quiet = False

    @staticmethod
    def mol_parse(mol: Union[str, list[str], Chem.Mol]) -> Chem.Mol:
        """
        parse molecule

        Parameters
        ----------
        mol: Chem.Mol

        Returns
        -------
        Chem.Mol
            return molecule.
        """
        if isinstance(mol, str):
            try:
                mol = Chem.MolFromSmiles(mol)
            except Exception:
                try:
                    mol = read_mol_db(mol)
                except Exception:
                    raise IOError("Could not parse " + str(mol))
        elif not isinstance(mol, Chem.Mol):
            raise ValueError("either list of str or rdmols")
        return mol

    @staticmethod
    def calc_n_rot_bonds(mol) -> int:
        """Function to calculate all rotatable bonds in a molecule (Element-H bonds are ignored).

        Returns
        -------
        int
           Number of rotatable bonds in a molecule.
        """
        rotatable_bonds_list = get_rotatable_bonds(mol)
        filtered_rotatable_bonds = filter_rotatable_bonds(rotatable_bonds_list)
        num_of_rot_bonds = len(filtered_rotatable_bonds)
        return num_of_rot_bonds

    @staticmethod
    def calc_num_heavy_atoms(mol) -> int:
        """Function to calculate all heavy atoms (non-H atoms) in a molecule.

        Returns
        -------
        int
           Total number of heavy atoms in a molecule
        """
        number_heavy_atoms = mol.GetNumHeavyAtoms()
        return number_heavy_atoms

    @staticmethod
    def calc_sulfur_present(mol) -> bool:
        """Function checks whether the sulfur atom is present in a molecule.

        Returns
        -------
        bool
            True if sulfur is present
        """

        atom_numbers = list(map(lambda x: x.GetAtomicNum(), mol.GetAtoms()))
        if 16 in atom_numbers:
            sulfur_present = True
        else:
            sulfur_present = False
        return sulfur_present

    @classmethod
    def calculate_mol_props(cls, mol: Chem.Mol) -> dict:
        """
        calculate properties of given mol.

        Parameters
        ----------
        mol: Chem.Mol

        Returns
        -------
        dict
            dict with properties.
        """

        mol = cls.mol_parse(mol)
        props = {}
        props["heavy_atoms"] = cls.calc_num_heavy_atoms(mol)
        props["rotational_bonds"] = cls.calc_n_rot_bonds(mol)
        props["sulfur_present"] = cls.calc_sulfur_present(mol)
        props["chem_formula"] = CalcMolFormula(mol)

        return props


    @staticmethod
    def get_interpolator_for_area(profile: pd.DataFrame, min_angle:float, max_angle:float) -> (
            Callable):
        """
            interpolator for range.

        Parameters
        ----------
        profile: pd.DataFrame
        min_angle: float
        max_angle: float

        Returns
        -------

        """
        x_angles = profile.angles.loc[(profile.angles >= min_angle) & (profile.angles <= max_angle)]
        y_energies = profile.rel_potential_energies.loc[
            (profile.angles >= min_angle) & (profile.angles <= max_angle)
        ]

        interpolator = metrics.interpolate_function(x_angles=x_angles, y_energies=y_energies)
        return interpolator

    @staticmethod
    def calc_difference_area(x_energies, y_energies, angles) -> float:
        """
            AUC difference metric

        Parameters
        ----------
        x_energies
        y_energies
        angles

        Returns
        -------

        """
        angles = angles + 180

        # deduplicate
        uangles = np.unique(angles)
        vi = [np.where(angles == i)[0][0] for i in uangles]
        angles = angles[vi]
        x_energies = x_energies[vi]
        y_energies = y_energies[vi]

        diff = np.abs(x_energies - y_energies)
        integral = integrate.simpson(diff, angles)
        return integral

    @staticmethod
    def _normalize_vector(vec: np.array) -> np.array:
        """
            normalize vector

        Parameters
        ----------
        vec

        Returns
        -------

        """
        maxV = vec.max()
        minV = vec.min()
        norm_vec = (vec - minV) / (maxV - minV)

        return norm_vec

    @classmethod
    def deal_with_diff_angle_ranges(
        cls, method_profile, ref_profile, min_angle=-178, max_angle=178
    ):
        """
        deal_with_diff_angle_ranges

        Parameters
        ----------
        method_profile
        ref_profile
        min_angle
        max_angle

        Returns
        -------

        """
        # fist two one tp envelops the other.

        if min(method_profile.angles) >= min(ref_profile.angles) and max(
            method_profile.angles
        ) <= max(
            ref_profile.angles
        ):  # ref envelopes or fits exact
            # print("env1")
            refmol_interpolator = metrics.interpolate_function(
                x_angles=ref_profile.angles,
                y_energies=ref_profile.rel_potential_energies,
                periodic_data=True,
            )

            angles = method_profile.angles
            method_profile_energies = method_profile.rel_potential_energies
            ref_profile_energies = refmol_interpolator(x=angles)

        elif min(ref_profile.angles) >= min(method_profile.angles) and max(
            ref_profile.angles
        ) <= max(
            method_profile.angles
        ):  # ref is subspace
            # print("env2")

            mol_interpolator = metrics.interpolate_function(
                x_angles=method_profile.angles, y_energies=method_profile.rel_potential_energies
            )

            angles = ref_profile.angles
            method_profile_energies = mol_interpolator(x=angles)
            ref_profile_energies = ref_profile.rel_potential_energies

        else:  # min max shifted offset of both tps.
            # print("env3")

            if min(method_profile.angles) <= min(ref_profile.angles) and max(
                method_profile.angles
            ) <= max(
                ref_profile.angles
            ):  # ref left trunc
                lower_angle_bound = min(ref_profile.angles)
                upper_angle_bound = max(method_profile.angles)
                # print("a")

            elif min(method_profile.angles) >= min(ref_profile.angles) and max(
                method_profile.angles
            ) >= max(
                ref_profile.angles
            ):  # ref right trunc
                # Case when the minimum angle is available for refmol and the maximum angle
                # is available for mol
                lower_angle_bound = min(method_profile.angles)
                upper_angle_bound = max(ref_profile.angles)
                # print("b")

            # log.warning("The interpolation of energies and thus all correlation values are
            # calculated within the range of: str(lower_angle_bound)  str(upper_angle_bound))

            ref_mol_interpolator = cls.get_interpolator_for_area(
                profile=ref_profile, min_angle=lower_angle_bound, max_angle=upper_angle_bound
            )

            mol_interpolator = cls.get_interpolator_for_area(
                profile=method_profile, min_angle=lower_angle_bound, max_angle=upper_angle_bound
            )

            angles = [
                i
                for i in set(ref_profile.angles).union(set(method_profile.angles))
                if i >= lower_angle_bound and i <= upper_angle_bound
            ]
            ref_profile_energies = ref_mol_interpolator(x=angles)
            method_profile_energies = mol_interpolator(x=angles)
        return np.array(angles), np.array(method_profile_energies), np.array(ref_profile_energies)

    @classmethod
    def calculate_torsionprofile_comparison_metrics(cls, method_profile, ref_profile):
        """

        Parameters
        ----------
        method_profile
        ref_profile

        Returns
        -------

        """
        angles, method_profile_energies, ref_profile_energies = cls.deal_with_diff_angle_ranges(
            method_profile, ref_profile
        )
        norm_refmol_energies = cls._normalize_vector(ref_profile_energies)
        norm_method_energies = cls._normalize_vector(method_profile_energies)
        if len(angles) > 5:
            mae = np.mean(np.abs(method_profile_energies - ref_profile_energies))
            rmse = np.sqrt(np.mean(np.square(method_profile_energies - ref_profile_energies)))
            rmse_bweight = np.sqrt(
                np.mean(
                    np.square(
                        metrics.get_boltzman_p(method_profile_energies - ref_profile_energies)
                        * (method_profile_energies - ref_profile_energies)
                    )
                )
            )
            crmse = metrics.calculate_coarse_rmse(
                ref_v=ref_profile_energies, method_v=method_profile_energies
            )

            # Hodgkin Measure
            hodgkin_measure = (2*np.sum(method_profile_energies * ref_profile_energies)) / np.sum(
                np.multiply(*np.square([method_profile_energies, ref_profile_energies])))

            # Carbo Metric
            norm_overlap_int = np.sum(method_profile_energies * ref_profile_energies) / np.sqrt(
                np.sum(np.square(method_profile_energies)) * np.sum(np.square(
                    ref_profile_energies)))

            norm_mae = np.mean(np.abs(norm_method_energies - norm_refmol_energies))
            norm_rmse = np.sqrt(np.mean(np.square(norm_method_energies - norm_refmol_energies)))
            # fd = sm.frechet_dist(x, y)

            min_shift = metrics.minima_shift_distance(method_profile_energies, ref_profile_energies)
            # Area dist
            area = cls.calc_difference_area(
                x_energies=ref_profile_energies, y_energies=method_profile_energies, angles=angles
            )
            norm_area = cls.calc_difference_area(
                x_energies=norm_refmol_energies, y_energies=norm_method_energies, angles=angles
            )

            # Dist Distribution
            wd_inv = metrics.earth_mover_distance(
                angles=angles, v_x=ref_profile_energies, v_y=method_profile_energies
            )
            wd_bolz = metrics.earth_mover_distance(
                angles=angles,
                v_x=ref_profile_energies,
                v_y=method_profile_energies,
                boltzmann_weigthing=True,
                inverse_energy=False,
            )
            wd = metrics.earth_mover_distance(
                angles=angles,
                v_x=ref_profile_energies,
                v_y=method_profile_energies,
                boltzmann_weigthing=False,
                inverse_energy=False,
            )

            # Corr
            r_spear = np.round(
                metrics.spearman_correlation(v_x=ref_profile_energies, v_y=method_profile_energies),
                2,
            )
            r_pears = np.round(
                metrics.pearson_correlation(v_x=ref_profile_energies, v_y=method_profile_energies),
                2,
            )

            raw_dict = {
                "mae": mae,
                "rmse": rmse,
                "crmse": crmse,
                "hodgkin": hodgkin_measure,
                "norm_overlap_int": norm_overlap_int,
                "norm_mae": norm_mae,
                "norm_rmse": norm_rmse,
                "boltzWeight_rmse": rmse_bweight,
                "diff_area": area,
                "norm_diff_area": norm_area,
                "r_spear": r_spear,
                "r_pears": r_pears,
                "wasserstein": wd,
                "wasserstein_inv": wd_inv,
                "wasserstein_boltzWeight": wd_bolz,
                "min_shift_d": min_shift,
                "angles": angles,
                "method_V": method_profile_energies,
                "ref_V": ref_profile_energies,
            }.items()

        else:
            raw_dict = {
                "mae": -1,
                "rmse": -1,
                "crmse": -1,
                "hodgkin": -1,
                "norm_overlap_int": -1,
                "norm_mae": -1,
                "norm_rmse": -1,
                "boltzWeight_rmse": -1,
                "diff_area": -1,
                "norm_diff_area": -1,
                "r_spear": -1,
                "r_pears": -1,
                "wasserstein": -1,
                "wasserstein_inv": -1,
                "wasserstein_boltzWeight": -1,
                "min_shift_d": -1,
                "angles": angles,
                "method_V": method_profile_energies,
                "ref_V": ref_profile_energies,
            }.items()

        return {k: np.round(v, 2) for k, v in raw_dict}

    """
    Structure Comparison Metrics:
    """

    @staticmethod
    def calculate_rmsd(method_mol, ref_mol) -> dict[str, int]:
        """
        Calculate the RMSD between two profiles.

        Parameters
        ----------
        method_mol
        ref_mol

        Returns
        -------

        """
        ref_torsion_atom_ids = tuple(map(int, ref_mol.GetProp("torsion_atom_ids").split()))
        rmsd_list = []
        mol_copy = Chem.Mol(method_mol)
        refmol_copy = Chem.Mol(ref_mol)

        mol_copy.RemoveAllConformers()
        refmol_copy.RemoveAllConformers()

        for i in range(method_mol.GetNumConformers()):
            mol_conf = method_mol.GetConformer(i)
            mol_angle = Chem.rdMolTransforms.GetDihedralDeg(
                conf=mol_conf,
                iAtomId=ref_torsion_atom_ids[0],
                jAtomId=ref_torsion_atom_ids[1],
                kAtomId=ref_torsion_atom_ids[2],
                lAtomId=ref_torsion_atom_ids[3],
            )

            for j in range(ref_mol.GetNumConformers()):
                refmol_conf = ref_mol.GetConformer(j)
                refmol_angle = Chem.rdMolTransforms.GetDihedralDeg(
                    conf=refmol_conf,
                    iAtomId=ref_torsion_atom_ids[0],
                    jAtomId=ref_torsion_atom_ids[1],
                    kAtomId=ref_torsion_atom_ids[2],
                    lAtomId=ref_torsion_atom_ids[3],
                )

                if round(mol_angle) == round(refmol_angle):
                    mol_copy.AddConformer(method_mol.GetConformer(i))
                    refmol_copy.AddConformer(ref_mol.GetConformer(j))
                    rmsd_i = round(Chem.rdMolAlign.GetBestRMS(mol_copy, refmol_copy), 5)
                    rmsd_list.append(rmsd_i)

                    mol_copy.RemoveAllConformers()
                    break
                refmol_copy.RemoveAllConformers()

        rmsd_average = np.array(rmsd_list).mean()
        rmsd_max = np.array(rmsd_list).max()
        return {"rmsd_average": rmsd_average, "rmsd_max": rmsd_max}

    """
    timing:
    """

    @staticmethod
    def calc_time(calculation_path: str) -> dict[str, int]:
        """This function calculates the time (average) per one conformer that was needed for
        this scan.

        Returns
        -------
        dict
            dict containing average, min and max timings.
        """
        orig_dir = bash.getcwd()
        bash.chdir(calculation_path)
        error_files = bash.glob("*/*err")

        timings = []
        if len(error_files) != 0:
            for file in error_files:
                f = open(file, "r")
                lines = f.readlines()
                for line in lines:
                    if "100%" in line:
                        try:
                            time_per_iter = float(line.split()[-1].replace("s/it]", ""))
                        except:
                            time_per_iter = float(line.split()[-1].replace("it/s]", ""))
                        timings.append(time_per_iter)
                        break

            time_averaged = np.array(timings).mean() / 60.0
            time_minimum = np.array(timings).min() / 60.0
            time_maximum = np.array(timings).max() / 60.0

            bash.chdir(orig_dir)
            timings = {
                "duration_average": time_averaged,
                "duration_min": time_minimum,
                "duration_max": time_maximum,
            }
            return timings
        else:
            raise ValueError("There are no .err files in this root directory!!!")

    def build_comparison_descriptors(self, method_profiles, ref_profiles) -> pd.DataFrame:
        """
        build the comparison descriptors

        Parameters
        ----------
        method_profiles
        ref_profiles

        Returns
        -------

        """
        not_found_in_both = 0
        collected_comparisons = []
        unique_molIds = ref_profiles.molID.unique()

        if self._quiet:
            iterator = unique_molIds
        else:
            iterator = tqdm(unique_molIds, desc="featurize")

        for molID in iterator:
            ref_data = ref_profiles.where(ref_profiles.molID == molID).dropna()
            method_data = method_profiles.where(method_profiles.molID == molID).dropna()

            if len(method_data) == 0 or len(ref_data) == 0:
                not_found_in_both += 1
                continue
            else:
                collected_metrics = {"molID": molID}
                mol_props = self.calculate_mol_props(ref_data.molecule.iloc[0])
                torsion_comparison = self.calculate_torsionprofile_comparison_metrics(
                    method_profile=method_data, ref_profile=ref_data
                )
                collected_metrics.update(mol_props)
                collected_metrics.update(torsion_comparison)
                collected_comparisons.append(collected_metrics)

        if not self._quiet:
            print("Did not find: " + str(not_found_in_both) + "/" + str(len(unique_molIds)))
        return pd.DataFrame(collected_comparisons)


    @staticmethod
    def normalize(
        metrics_df: pd.DataFrame, normer=MinMaxScaler(), remove_outlier=True, outlier_lim=1.5
    ) -> pd.DataFrame:
        """
        normalize a data set

        Parameters
        ----------
        metrics_df
        normer
        remove_outlier
        outlier_lim

        Returns
        -------

        """
        if remove_outlier:
            # Cut data Outliers
            for col in metrics_df.columns:
                limit = metrics_df[col].mean() + metrics_df[col].std() * outlier_lim
                metrics_df[col] = metrics_df[col].clip(upper=limit)

        normalized_data = normer.fit_transform(metrics_df)
        if isinstance(metrics_df, pd.DataFrame):
            normalized_df = pd.DataFrame(
                {col_name: col for col_name, col in zip(metrics_df.columns, normalized_data.T)}
            )
        return normalized_df

    def dimension_reduction(
        self,
        normalized_df: pd.DataFrame,
        test_max_n_components: int = 10,
        explain_variance_limit: float = 0.85,
        pc_explain_at_least: float = 0.1,
    ) -> np.array:
        """
        Reduce the dimensionality of a given metric space.

        Parameters
        ----------
        normalized_df
        test_max_n_components
        explain_variance_limit
        pc_explain_at_least

        Returns
        -------

        """
        # Dim Reduction:
        ## build initial model:
        test_max_n_components = (
            test_max_n_components
            if (normalized_df.shape[1] > test_max_n_components)
            else normalized_df.shape[1]
        )
        self.pca_model_test = PCA(n_components=test_max_n_components)
        self.pca_model_test.fit(normalized_df)

        ## cut on 10% explaining variance.
        larger_pcs = self.pca_model_test.explained_variance_ratio_[
            np.where(self.pca_model_test.explained_variance_ratio_ > pc_explain_at_least)
        ]
        if np.round(sum(larger_pcs), 2) > explain_variance_limit:
            reasonable_components = len(larger_pcs)
        else:
            raise Exception(
                "Did not find enough reasonable pc or only to many!"
                + str(self.pca_model_test.explained_variance_ratio_)
                + "\nSum: "
                + str(np.round(sum(larger_pcs), 2))
            )

        ## build real model:
        self.pca_model = PCA(n_components=reasonable_components)
        self.pca_model.fit(normalized_df)

        # transform data:
        pc_data = self.pca_model.transform(normalized_df)

        print(
            "Found following pcs with variance explainability: ",
            np.round(self.pca_model.explained_variance_ratio_, 2),
        )
        return pc_data

    def _k_mean_elbow_method(self, scaled_data, k_range, alpha_k=0.02):
        """
        Try to find best k-means cluster number by inertia/cluster efficiency
        Source:
        https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means
         -c28e614ecb2c

        """
        ans = []
        inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()

        for k in k_range:
            kmeans_clustering = KMeans(n_clusters=k, random_state=0, n_init="auto")
            clustering = kmeans_clustering.fit(scaled_data)
            scaled_inertia = clustering.inertia_ / inertia_o + alpha_k * k
            ans.append((k, np.round(scaled_inertia, 2)))

        inertias = pd.DataFrame(ans, columns=["k", "Scaled Inertia"]).set_index("k")
        best_k = inertias.idxmin()[0]
        return best_k, inertias

    def k_means_clustering(
        self, data: np.array, k_cluster_range=range(3, 20), ncluster_penalty: float = 0.02
    ):
        """
        Cluster with k-means

        Parameters
        ----------
        data
        k_cluster_range
        ncluster_penalty

        Returns
        -------

        """
        k_clusters, self.k_means_inertias = self._k_mean_elbow_method(
            data, k_cluster_range, alpha_k=ncluster_penalty
        )
        print("optimal cluster number:", k_clusters)

        kmeans_clustering = KMeans(n_clusters=k_clusters, random_state=0, n_init="auto")
        self.clustering_model = kmeans_clustering.fit(data)

        c_center = self.clustering_model.cluster_centers_
        c_id = self.clustering_model.predict(data)
        return c_center, c_id

    def _build_decoy_data(
        self,
        reference_df,
        scale_max=4,
        scale_step=0.5,
        iscale_max=4,
        iscale_step=0.5,
        shift_max=90,
        shift_step=10,
    ):
        """
        build up a decoy data set.

        Parameters
        ----------
        reference_df
        scale_max
        scale_step
        iscale_max
        iscale_step
        shift_max
        shift_step

        Returns
        -------

        """
        reference_data = deepcopy(reference_df)

        def shif_angle_periodically(angle, shift):
            new_angle = angle + shift

            if new_angle > 180:
                new_angle = -180 + (new_angle % 180)
            elif new_angle < -180:
                new_angle = 180 - (np.abs(new_angle) % 180)
            else:
                pass
            return new_angle

        # Augment data:
        shift_scaling_differences = []
        for shift in tqdm(np.arange(-shift_max, shift_max, shift_step), desc="BuildDecoyData"):
            for scale in np.arange(1, scale_max, scale_step):
                diff_df = deepcopy(pd.DataFrame(reference_data))

                scaled_V = diff_df.rel_potential_energies * scale
                diff_df.rel_potential_energies = scaled_V

                diff_df.angles = diff_df.angles.apply(lambda x: shif_angle_periodically(x, shift))

                for molID in diff_df.molID.unique():
                    ref_data = pd.DataFrame(reference_data.loc[reference_data.molID == molID])
                    diff_data = pd.DataFrame(diff_df.loc[diff_df.molID == molID])
                    diff_data.sort_values(by="angles", inplace=True)
                    ref_data.sort_values(by="angles", inplace=True)

                    crmse = metrics.calculate_coarse_rmse(
                        ref_v=ref_data.rel_potential_energies.to_numpy(),
                        method_v=diff_data.rel_potential_energies.to_numpy(),
                    )

                    if scale == 1 and shift == 0:
                        diff_df.loc[diff_df.molID == molID, "approach"] = "perfect"
                        diff_df.loc[diff_df.molID == molID, "mut"] = "perfect"
                        diff_df.loc[diff_df.molID == molID, "class"] = "good"

                    elif np.abs(shift) == 0:
                        diff_df.loc[diff_df.molID == molID, "approach"] = "scale_" + str(scale)
                        diff_df.loc[diff_df.molID == molID, "mut"] = "scale_" + str(scale)

                        if np.abs(scale) < 2.25:  # crmse<0.6): #
                            diff_df.loc[diff_df.molID == molID, "class"] = "good"
                        elif np.abs(scale) < 3.5:  # crmse<1.25): #
                            diff_df.loc[diff_df.molID == molID, "class"] = "medium"
                        else:
                            diff_df.loc[diff_df.molID == molID, "class"] = "bad"

                    elif scale == 1:
                        diff_df.loc[diff_df.molID == molID, "approach"] = "shift_" + str(shift)
                        diff_df.loc[diff_df.molID == molID, "mut"] = "shift_" + str(shift)

                        if np.abs(shift) < 35:
                            diff_df.loc[diff_df.molID == molID, "class"] = "good"
                        elif np.abs(shift) < 45:
                            diff_df.loc[diff_df.molID == molID, "class"] = "medium"
                        else:
                            diff_df.loc[diff_df.molID == molID, "class"] = "bad"
                    else:
                        diff_df.loc[diff_df.molID == molID, "approach"] = (
                            "mix_" + str(scale) + "_" + str(shift)
                        )
                        diff_df.loc[diff_df.molID == molID, "mut"] = (
                            "mix_" + str(scale) + "_" + str(shift)
                        )

                        if np.abs(shift) < 35 and crmse < 0.6:  # np.abs(scale)<3.5):
                            diff_df.loc[diff_df.molID == molID, "class"] = "good"
                        elif np.abs(shift) < 45 and crmse < 1.25:  # np.abs(scale)<4.0):
                            diff_df.loc[diff_df.molID == molID, "class"] = "medium"
                        else:
                            diff_df.loc[diff_df.molID == molID, "class"] = "bad"

                shift_scaling_differences.append(diff_df)

        shift_inv_scaling_differences = []
        for shift in np.arange(-shift_max, shift_max, shift_step):
            for scale in np.arange(-iscale_max, iscale_max, iscale_step):
                diff_df = deepcopy(pd.DataFrame(reference_data))

                scaled_V = (
                    diff_df.rel_potential_energies + np.exp(-diff_df.rel_potential_energies) * scale
                )
                diff_df.rel_potential_energies = scaled_V

                diff_df.angles = diff_df.angles.apply(lambda x: shif_angle_periodically(x, shift))

                for molID in diff_df.molID.unique():
                    ref_data = pd.DataFrame(reference_data.loc[reference_data.molID == molID])
                    diff_data = pd.DataFrame(diff_df.loc[diff_df.molID == molID])
                    ref_data = ref_data.sort_values(by="angles")
                    diff_data = diff_data.sort_values(by="angles")

                    crmse = metrics.calculate_coarse_rmse(
                        ref_v=ref_data.rel_potential_energies.to_numpy(),
                        method_v=diff_data.rel_potential_energies.to_numpy(),
                    )

                    if shift == 0:
                        diff_df["approach"] = "iscale_" + str(scale)
                        diff_df["mut"] = "iscale_" + str(scale)
                        if crmse < 0.6:  # np.abs(scale)<1.75):
                            diff_df["class"] = "good"
                        elif crmse < 1.25:  # np.abs(scale)<2.25):
                            diff_df["class"] = "medium"
                        else:
                            diff_df["class"] = "bad"
                    else:
                        diff_df["approach"] = "mix_i" + str(scale) + "_" + str(shift)
                        diff_df["mut"] = "mix_i" + str(scale) + "_" + str(shift)
                        if np.abs(shift) < 35 and crmse < 0.6:  # np.abs(scale)<1.5):
                            diff_df["class"] = "good"
                        elif np.abs(shift) < 45 and crmse < 1.25:  # np.abs(scale)<2):
                            diff_df["class"] = "medium"
                        else:
                            diff_df["class"] = "bad"
                    shift_scaling_differences.append(diff_df)

        arti_dfs = shift_scaling_differences + shift_inv_scaling_differences
        # artificial_dfs = pd.concat(arti_dfs)

        return arti_dfs

    def _compare_all_approaches(self, reference_df, method_dfs):
        """
        compare the different approaches.

        Parameters
        ----------
        reference_df
        method_dfs

        Returns
        -------

        """
        orig_quiet = self._quiet
        self._quiet = True
        nbatches = (
            self.n_processes * 10 if (self.n_processes * 10 < len(method_dfs)) else len(method_dfs)
        )
        ndatas = len(method_dfs) - 1
        batch_size = ndatas // nbatches
        n_rest = ndatas % nbatches
        batch_sizes = [batch_size + 1 if (i < n_rest) else batch_size for i in range(nbatches)]
        batch_data_range = [
            list(range(sum(batch_sizes[:i]), sum(batch_sizes[:i]) + batch_size + 1))
            for i in range(len(batch_sizes))
        ]

        p = mult.Pool(self.n_processes)
        ta_ks = vars(self)

        distribute_jobs = [
            (n, reference_df, method_dfs, batch_data_range[n], ta_ks) for n in range(nbatches)
        ]
        p_job_res = p.starmap(
            thread_calculate_distances,
            tqdm(distribute_jobs, total=nbatches, desc="CalculateMetrics: "),
        )
        p.close()
        p.terminate()
        p.join()

        self._quiet = orig_quiet

        comparison_ds = pd.concat([p for pl in p_job_res for p in pl])

        return comparison_ds

    def _build_decoy_dist(
        self,
        reference_df,
        scale_max=4,
        scale_step=0.5,
        iscale_max=4,
        iscale_step=0.5,
        shift_max=90,
        shift_step=10,
    ):
        """
        build up a decoy metric space.

        Parameters
        ----------
        reference_df
        scale_max
        scale_step
        iscale_max
        iscale_step
        shift_max
        shift_step

        Returns
        -------

        """
        decoy_data = self._build_decoy_data(
            reference_df=reference_df,
            scale_max=scale_max,
            scale_step=scale_step,
            iscale_max=iscale_max,
            iscale_step=iscale_step,
            shift_max=shift_max,
            shift_step=shift_step,
        )

        # calculate distances:
        orig_quiet = self._quiet
        self._quiet = True
        nbatches = (
            self.n_processes * 10 if (self.n_processes * 10 < len(decoy_data)) else len(decoy_data)
        )
        ndatas = len(decoy_data) - 1
        batch_size = ndatas // nbatches
        n_rest = ndatas % nbatches
        batch_sizes = [batch_size + 1 if (i < n_rest) else batch_size for i in range(nbatches)]
        batch_data_range = [
            list(range(sum(batch_sizes[:i]), sum(batch_sizes[:i]) + batch_size + 1))
            for i in range(len(batch_sizes))
        ]

        if self.n_processes > 1:
            p = mult.Pool(self.n_processes)
            ta_ks = vars(self)

            distribute_jobs = [
                (n, reference_df, decoy_data, batch_data_range[n], ta_ks) for n in range(nbatches)
            ]
            p_job_res = p.starmap(
                thread_calculate_distances,
                tqdm(distribute_jobs, total=nbatches, desc="CalculateMetrics: "),
            )
            p.close()
            p.terminate()
            p.join()
            arificial_ds = pd.concat([p for pl in p_job_res for p in pl])

        else:
            ta_ks = vars(self)
            p_job_res = thread_calculate_distances(
                0, reference_df, decoy_data, batch_data_range[0], ta_ks
            )
            arificial_ds = pd.concat([p for p in p_job_res])
        self._quiet = orig_quiet

        return arificial_ds, decoy_data

    def _classify_based_on_decoy_data(
        self, ds, good_clusterers=["perfect"], medium_clusterers=["scale_2.0", "shift_10"]
    ):
        """
        classify decoy data.

        Parameters
        ----------
        ds
        good_clusterers
        medium_clusterers

        Returns
        -------

        """
        good = list(ds.where(ds.method_type.isin(good_clusterers)).cluster_id.dropna().unique())

        medium = ds.where(ds.method_type.isin(medium_clusterers)).cluster_id.dropna().unique()
        medium = list(set(medium).difference(good))

        bad = (
            ds.where(~ds.method_type.isin(good_clusterers + medium_clusterers))
            .cluster_id.dropna()
            .unique()
        )
        bad = list(set(bad).difference(good).difference(medium))

        classification = {"C2": good, "C1": medium, "C3": bad}

        return classification
