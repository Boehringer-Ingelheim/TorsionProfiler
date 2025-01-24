# This code is part of TorsionProfiler and is licensed under the MIT license.
# For details, see https://github.com/bi/torsionprofiler

"""
This Module implements the parameters for the CLI
"""

from .mol import MOL
from .mol_dir import MOL_DIR

from .output import OUTPUT_FILE_AND_EXT
from .output_dir import OUTPUT_DIR

from .torsion_params import (TORSIONATOMS,
                             ALLTORSIONATOMS, ALLFRAGMENTEDTORSIONATOMS,
                             NMEASUREMENT, TORSIONATOMSB)

from .orchestration import SUBMITSLURM as SUBMIT
