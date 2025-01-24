# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
This file implements the torsoin atom parameters.
"""

import ast
from plugcli.params import MultiStrategyGetter, Option


def _load_torsion_ids_from_cmd(user_input, context):
    """
    translate a provided string to the torsion atom id tuple
    """
    if "[" in user_input or "(" in user_input:
        return tuple(map(lambda x: int(x) - 1, ast.literal_eval(user_input)))

    if "," in user_input:
        return tuple(map(lambda x: int(x) - 1, user_input.split(",")))

    return tuple(map(lambda x: int(x) - 1, user_input.split()))


get_torsion_atom_ids = MultiStrategyGetter(
    strategies=[
        _load_torsion_ids_from_cmd,
    ],
    error_message="Unable to generate a molecule from '{user_input}'.",
)

TORSIONATOMS = Option(
    "-t",
    "--torsionIDs",
    help=(
        'atom ids that describe a torsion ("(1,2,3,4)" or 1,2,3,4) as string. Atom Ids start at 1!'
    ),
    getter=get_torsion_atom_ids,
    required=False,
)

TORSIONATOMSB = Option(
    "-tb",
    "--torsionIDsB",
    help=(
        'atom ids that describe a torsion ("(1,2,3,4)" or 1,2,3,4) as string. Atom Ids start at 1!'
    ),
    getter=get_torsion_atom_ids,
    required=False,
    default=None,
)

ALLTORSIONATOMS = Option(
    "-a",
    "--all-torsions",
    help=("This flag can be used instead of --torsion"),
    default=False,
    count=False,
)

ALLFRAGMENTEDTORSIONATOMS = Option(
    "-af",
    "--all-fragmented-torsions",
    help=("This flag can be used instead of --torsion"),
    default=False,
    count=False,
)


NMEASUREMENT = Option(
    "-n",
    "--n-measurements",
    help=(
        "how many calculations shall be done on the profile? "
        "\t 37 is fine leading to 10 degree steps."
        "\t 24 is a bit coarser."
    ),
    default=24,
    required=False,
)
