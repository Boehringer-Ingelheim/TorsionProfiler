#!/usr/bin/env python3

#IMPORTS
from datetime import datetime
start=datetime.now()
from rdkit import Chem
from torsion_profiler.engines.mmff94_calculator import Mmff94Calculator


#VARIABLES: 
mols = [m for m in Chem.SDMolSupplier("torsion_profiler_cli_testsruo4y0c6/tmp_test_opt_cli_1d_local_Mmff94Calculator79uqisi1/work/tmp_work_sp7ufan_wx_Mmff94Calculator_0/in_work_singlepoint_mol.sdf", removeHs=False)]
mol = mols[0]
if(len(mols)>1): [mol.AddConformer(m.GetConformer(), i) for i,m in enumerate(mols[1:], start=1)]
out_file_path = "torsion_profiler_cli_testsruo4y0c6/tmp_test_opt_cli_1d_local_Mmff94Calculator79uqisi1/work/tmp_work_sp7ufan_wx_Mmff94Calculator_0/work_opt_Mmff94Calculator_0_singlepoint.sdf"



#DO
target_obj = Mmff94Calculator(ff_model="MMFF94", optimize_structure=False, optimize_structure_nsteps=100000, optimize_structure_tol=1e-06, optimize_structure_write_out=True, force_constant=10000)

target_obj.calculate_conformer_potentials(mol = mol, out_file_path = out_file_path, )

end=datetime.now()
duration=end-start
print("Duration: ",duration.seconds, "s\n")

exit(0)
