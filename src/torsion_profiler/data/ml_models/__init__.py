"""
all the test torsionprofile are in here.
"""
from ...utils import bash

root_path = __path__[0]

mace_model_root = f"{root_path}/mace-off/mace_off23"
print(mace_model_root)
mace_models = {bash.path.basename(k).replace(".model", ""): k
               for k in bash.glob(f"{mace_model_root}/*.model")
               }