"""
Chemical Metrics
"""
from typing import Iterable, Callable

import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
from scipy.spatial.distance import jaccard
from sklearn import neighbors

from . import units


# function Fitting
def learn_function(
    x_angles: Iterable[float], y_energies: Iterable[float], n_neighbors: int = 2
) -> Callable:
    """
    This function can learn a torsion profile an help interpolating between calculated points.
    It will return a function, that can take another angle range and it will predict the according
    energy values.

    !Future: add more and different functionals.

    Parameters
    ----------
    x_angles : Iterable[float]
        angles, used to calculate energies
    y_eneriges : Iterable[float]
        energies fitting to the x_angles, calculated
    n_neighbors : int, optional
        n_neighbors for KNRegressor method, by default 5

    Returns
    -------
    function
        function, that can be used to interpolate the torsion profile on a new angle distribution
        (new angle distribution must be in the range of original angles!)
    """
    sk_x = np.array(x_angles, ndmin=2).T
    mod = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
    _ = mod.fit(sk_x, y_energies)

    # return a function, giving the interpolated values
    def fitter(x: Iterable[float]) -> Iterable[float]:
        if min(x) >= min(x_angles) and max(x) <= max(x_angles):
            energies = mod.predict(np.array(x, ndmin=2).T)
        else:
            raise ValueError(
                "new Angles must be in the range of fitted ones! Ranges: min="
                + str(min(x_angles))
                + " max="
                + str(max(x_angles))
            )
        return energies

    return fitter


def interpolate_function(
    x_angles: Iterable[float],
    y_energies: Iterable[float],
    periodic_data: bool = True,
    allow_out_of_range: bool = True,
) -> Callable:
    """
    This function interpolates inbetween provided datapoints..

    Parameters
    ----------
    x_angles : Iterable[float]
        angles, used to calculate energies
    y_eneriges : Iterable[float]
        energies fitting to the x_angles, calculated
    periodic_data: bool, optional
        assume periodic data (def. True)
    allow_out_of_range : int, optional
        if a data point  is out of range, if true fill up with closest data point, else raise Error.

    Returns
    -------
    function
        interpolating function.
    """

    # return a function, giving the interpolated values
    def fitter(x: Iterable[float]) -> Iterable[float]:
        if (min(x) >= min(x_angles) and max(x) <= max(x_angles)) or allow_out_of_range:
            if periodic_data:
                energies = np.interp(x=x, xp=x_angles, fp=y_energies, period=360)
            else:
                energies = np.interp(x=x, xp=x_angles, fp=y_energies, period=periodic_data)
        else:
            raise ValueError(
                "new Angles must be in the range of fitted ones! Ranges: min="
                + str(min(x_angles))
                + " max="
                + str(max(x_angles))
            )
        return energies

    return fitter


# Classifications
def get_boltzman_p(d_e: Iterable[float], t: float = 298.0) -> Iterable[float]:
    """
        give boltzman weight of given energy difference.
    Parameters
    ----------
    d_e: Iterable[float]
        energy differences, in units of kJ/mol
    t: float
        temperature

    Returns
    -------
    Iterable[float]
        probabilities
    """
    beta = 1000 / (units.k_b * units.avogadro_constant * t)
    return np.exp(-beta * d_e)


def get_boltzman_ensemble_p(d_e: Iterable[float], t: float = 298.0, n=None,
                            pseudo_count:float=0.0000001) -> Iterable[float]:
    """
        Under the Assumtption, that dE describes a full ensemble, return relative ps
    Parameters
    ----------
    d_e: Iterable[float]
        energy differences, in units of kJ/mol
    t: float
        temperature

    Returns
    -------
    Iterable[float]
        relative probabilities
    """

    if n is None:
        n = len(d_e)
    beta = 1000 / (units.k_b * units.avogadro_constant * t)
    ps = stats.boltzmann.sf(d_e, lambda_=beta, N=n)
    normed_p = ps / (sum(ps)+pseudo_count)
    return normed_p


def get_relative_energy_classifications(vs: Iterable[float]) -> list[str]:
    """
    relative energy classifications

    Parameters
    ----------
    vs: Iterable[float]
        input potential energies

    Returns
    -------
    list[str]
        classifications
    """
    mean = np.mean(vs)
    std = np.std(vs)
    general_scale = {0: "green", mean - std: "gold", mean: "orange", mean + std: "red"}
    colors = [[general_scale[k] for k in sorted(general_scale) if k <= v][-1] for v in vs]
    return colors


def get_simple_energy_classifications(vs: Iterable[float]) -> list[str]:
    """
    simple energy classifications

    Parameters
    ----------
    vs: Iterable[float]
        input potential energies

    Returns
    -------
    list[str]
        classifications
    """
    general_scale = {0: "green", 2.5: "gold", 4: "orange", 6: "red"}
    colors = []
    for v in vs:
        c = None
        for k in sorted(general_scale):
            if k <= v:
                c = general_scale[k]

        if c is None:
            c = general_scale[6]

        colors.append(c)
    return colors


def calculate_coarse_rmse(ref_v: Iterable[float], method_v: Iterable[float]) -> float:
    """
        Calculate a coarse RMSE that tries to caputer flexibility categorizations.

    Parameters
    ----------
    ref_v: Iterable[float]
    method_v: Iterable[float]

    Returns
    -------

    """
    categories = {"green": 0, "gold": 0.5, "orange": 1, "red": 1.25}

    classes_ref = np.array([categories[c] for c in get_simple_energy_classifications(ref_v)])
    classes_method = np.array([categories[c] for c in get_simple_energy_classifications(method_v)])

    crmse = np.sqrt(np.mean(np.square(classes_ref - classes_method)))
    return crmse


# Correlations of energies:


def pearson_correlation(v_x: Iterable[float], v_y: Iterable[float]) -> float:
    """
    This function calculates the pearson correlation for two energy distributions.
    You need to make sure that both distributions use the same angle values in the same sequence!

    Parameters
    ----------
    v_x : Iterable[float]
        energies by method X
    v_y : Iterable[float]
        energies method by method Y

    Returns
    -------
    float
        r-value
    """
    return stats.pearsonr(v_x, v_y)[0]


def spearman_correlation(v_x: Iterable[float], v_y: Iterable[float]) -> float:
    """
    This function calculates the spearman correlation for two energy distributions.
    You need to make sure that both distributions use the same angle values in the same sequence!

    Parameters
    ----------
    v_x : Iterable[float]
        energies by method X
    v_y : Iterable[float]
        energies method by method Y

    Returns
    -------
    float
        r-value
    """
    return stats.spearmanr(v_x, v_y)[0]


# Shape of curves:


def minima_shift_distance(v_x, v_y, position_blurr: int = 3) -> float:
    """
    This function calculates a normalised bitwise minima shift distance.
    first the minima are determined and a bitvector with the minima indicated by their p.
    Next the two generated bitvectors are used to calculate a jaccard distance:
        (c(TF)+c(FT))/(c(TT)+c(FT)+c(TF))


    Parameters
    ----------
    v_x: Iterable[float],
    v_y: Iterable[float],

    position_blurr: int, optional
        this blurrs the position of the minima (3 == +-1 position in the vector)

    Returns
    -------
    distance : float
        the calculated distance
    """
    minima1 = argrelextrema(v_x, np.less, mode="wrap")[0]
    minima2 = argrelextrema(v_y, np.less, mode="wrap")[0]

    fp1 = np.zeros(len(v_x))
    fp1[minima1] = 1

    fp2 = np.zeros(len(v_y))
    fp2[minima2] = 1

    cfp1 = np.convolve(fp1, np.ones(position_blurr), "valid")
    cfp2 = np.convolve(fp2, np.ones(position_blurr), "valid")
    d = jaccard(cfp1, cfp2)

    return d


def earth_mover_distance(
    angles: Iterable[float],
    v_x: Iterable[float],
    v_y: Iterable[float],
    inverse_energy: bool = True,
    c: float = 0.00000001,
    boltzmann_weigthing: bool = False,
    boltzmann_weigthing_t: float = 298,
) -> float:
    """
    This metric is called the Wasserstein distance or Earth-mover distance.
    It calculates the distance two distributions by their shape overlap / probability overlap.
    there are three possible weighting methods of the distributions.
    Methods:
    * no weighting (inverse_energy and boltzmann_weigthing == False)
    * inverse energy weigthing
    * boltzman weighting

    Parameters
    ----------
    angles : Iterable[float]
        angle range used for both potentials
    v_x : Iterable[float]
        Energies calculated for the angles with method X
    v_y : Iterable[float]
        Energies calculated for the angles with method Y
    inverse_energy : bool, optional
        Use inverse energy Weighting (this leads to more weights on minima compared to
        curve maxima), by default True
    c : float, optional
        pseudo-count to inhibit division by zero singularities, by default 0.00000001
    boltzmann_weigthing : bool, optional
        the distribution distance differences will be boltzmann weighted(assumes kcal/mol as input),
         by default False
    boltzmann_weigted_T : float, optional
        Temperature for the boltzman weighting - default is one to have a very sensitive weighting.,
        by default 1

    Returns
    -------
    float
        waterstein distance

    Raises
    ------
    ValueError
        if (inverse_energy and boltzmann_weigthing == True) because both methods can not be used
        at the same time in this implementation.
    """
    if boltzmann_weigthing and inverse_energy:
        raise ValueError("This does not work, either boltzmann weigted or inverse energies!")

    if inverse_energy and not boltzmann_weigthing:
        # weigthing = lambda x: 1/(x+c)
        # weigths_x = weigthing(V_x)
        weigths_x = list(map(lambda x: 1 / (np.abs(x) + c), v_x))
        norm_weighting_x = weigths_x / sum(weigths_x)
        # weigths_y = weigthing(V_y)
        weigths_y = list(map(lambda x: np.abs(1 / (np.abs(x) + c)), v_y))
        norm_weighting_y = weigths_y / sum(weigths_y)
        wd = stats.wasserstein_distance(
            u_values=angles, v_values=angles, u_weights=norm_weighting_x, v_weights=norm_weighting_y
        )

    elif boltzmann_weigthing:
        weights = np.abs(get_boltzman_p(d_e=-np.abs((v_x - v_y) * 4.184), t=boltzmann_weigthing_t))
        norm_weighting = np.nan_to_num(weights, posinf=100) / sum(
            np.nan_to_num(weights, posinf=100)
        )
        wd = stats.wasserstein_distance(u_values=angles, v_values=angles, u_weights=norm_weighting)
    else:
        try:
            wd = stats.wasserstein_distance(
                u_values=angles, v_values=angles, u_weights=np.abs(v_x), v_weights=np.abs(v_y)
            )
        except Exception:
            wd = -1
    return wd
