"""
Systematic data set
"""

import re

from rdkit import Chem

from ....utils import bash


class SystematicSet:
    """
    Get systematically generated data set
    """
    out_folder_prefix = bash.path.dirname(__file__)

    linkers = [
        "",
        "[CH2:2]",
        "[O:2]",
        "[NH:2]",
        "[S:2]",
        "[C:2](=O)",
        "[C:2](=O)[NH:3]",
        "[S:2](=O)(=O)",
        "[S:2](=O)(=O)[NH:3]",
        "[S:2](=O)(=O)[NH:3][C:4](=O)",
        "[S:2](=O)(=NC)[NH:3][C:4](=O)",
        "[S:2](=O)",
    ]

    carbon_scaffolds = [
        "C[CH2:1]X[CH2:3]C",
        "c1cccc[c:1]1X[CH2:3]C",
        "c1cccc[c:1]1X[c:3]2ccccc2",
    ]

    heteroCycle_scaffolds = [
        # 5 member Ring with 5 member Ring
        "c1cOc[c:1]1X[c:3]2cOcc2",
        "c1cOc[c:1]1X[c:3]2cNcc2",
        "c1cOc[c:1]1X[c:3]2cScc2",
        "c1cOc[c:1]1X[c:3]2Occc2",
        "c1cOc[c:1]1X[c:3]2Nccc2",
        "c1cOc[c:1]1X[c:3]2Sccc2",
        "c1ccO[c:1]1X[c:3]2Occc2",
        "c1ccO[c:1]1X[c:3]2Nccc2",
        "c1ccO[c:1]1X[c:3]2Sccc2",
        "c1cNc[c:1]1X[c:3]2cNcc2",
        "c1cNc[c:1]1X[c:3]2cScc2",
        "c1cNc[c:1]1X[c:3]2Nccc2",
        "c1cNc[c:1]1X[c:3]2Sccc2",
        "c1ccN[c:1]1X[c:3]2Nccc2",
        "c1ccN[c:1]1X[c:3]2Sccc2",
        "c1cSc[c:1]1X[c:3]2cScc2",
        "c1cSc[c:1]1X[c:3]2Sccc2",
        "c1ccS[c:1]1X[c:3]2Sccc2",
        # 5 member Ring with 6 member Ring
        "c1cNc[c:1]1X[c:3]2ccccc2",  # Pyrrol 1
        "c1ccN[c:1]1X[c:3]2ccccc2",  # Pyrrol 2
        "c1ccc[n:1]1X[c:3]2ccccc2",  # Pyrrol 3
        "c1ccS[c:1]1X[c:3]2ccccc2",  # Tiophen
        "c1cSc[c:1]1X[c:3]2ccccc2",  # Tiophen 2
        "c1ccO[c:1]1X[c:3]2ccccc2",  # Furan
        "c1cOc[c:1]1X[c:3]2ccccc2",  # Furan 2
        "c1cNn[c:1]1X[c:3]2ccccc2",  # Pyrazol 1
        "c1ccn[n:1]1X[c:3]2ccccc2",  # Pyrazol 2
        "n1cNc[c:1]1X[c:3]2ccccc2",  # Imidazol 1
        "n1ccN[c:1]1X[c:3]2ccccc2",  # Imidazol 1
        "c1ncc[n:1]1X[c:3]2ccccc2",  # Imidazol 2
        "c1ncS[c:1]1X[c:3]2ccccc2",  # Thiazol 1
        "n1ccS[c:1]1X[c:3]2ccccc2",  # Thiazol 2
        "c1ncO[c:1]1X[c:3]2ccccc2",  # Oxazol 1
        "n1ccO[c:1]1X[c:3]2ccccc2",  # Oxazol 2
        "c1cOc[c:1]1X[c:3]2cccnc2",
        "c1ccO[c:1]1X[c:3]2cccnc2",
        "c1ccO[c:1]1X[c:3]2ccccn2",
        "c1cSc[c:1]1X[c:3]2cccnc2",
        "c1ccS[c:1]1X[c:3]2cccnc2",
        "c1ccS[c:1]1X[c:3]2ccccn2",
        "c1cNc[c:1]1X[c:3]2cccnc2",
        "c1ccN[c:1]1X[c:3]2cccnc2",
        "c1ccN[c:1]1X[c:3]2ccccn2",
        "c1cNc[c:1]1X[c:3]2nccnc2",
        "c1ccN[c:1]1X[c:3]2nccnc2",
        # 6 member Rings with 6 member Rings
        # "C1CCCCN1X[c:3]2ccccc2", #Piperidine 1
        "c1cccn[c:1]1X[c:3]2ccccc2",  # Pyridin 2
        "c1cncn[c:1]1X[c:3]2ccccc2",  # Pyrimidin 1
        "n1cccn[c:1]1X[c:3]2ccccc2",  # Pyrimidin 2
        # double Ring with 6 member Rings
        "c13c(cccc3)nc[n:1]1X[c:3]2ccccc2",  # Benzimidazol
        "c13c(cccc3)cc[n:1]1X[c:3]2ccccc2",  # Indol 1
        "c13c(ccN3)ccc[c:1]1X[c:3]2ccccc2",  # Indol 2
        "c13c(ncnc3)nc[n:1]1X[c:3]2ccccc2",  # Purin 1
        "n1c3c(ncnc3)n[CH:1]1X[c:3]2ccccc2",  # Purin 2
    ]

    carbon_scaffolds_withSubst = [
        "C[C:1]YX[C:3]ZC",
        "CY[C:1]X[C:3]CZ",
        "c1ccccY[c:1]1X[C:3]ZC",
        "c1ccccY[c:1]1X[c:3]2cZcccc2",
    ]

    heteroCycle_Subst_scaffolds = [
        # 5 member Ring with 5 member Ring
        "c1cOcY[c:1]1X[c:3]2cZOcc2",
        "c1cOcY[c:1]1X[c:3]2cZNcc2",
        "c1cOcY[c:1]1X[c:3]2cZScc2",
        "c1cOcY[c:1]1X[c:3]2Occc2",
        "c1cOcY[c:1]1X[c:3]2NZccc2",
        "c1cOcY[c:1]1X[c:3]2Sccc2",
        "c1ccO[c:1]1X[c:3]2Occc2",
        "c1ccO[c:1]1X[c:3]2NZccc2",
        "c1ccO[c:1]1X[c:3]2Sccc2",
        "c1cNcY[c:1]1X[c:3]2cZNcc2",
        "c1cNcY[c:1]1X[c:3]2cZScc2",
        "c1cNcY[c:1]1X[c:3]2NZccc2",
        "c1cNcY[c:1]1X[c:3]2Sccc2",
        "c1ccNY[c:1]1X[c:3]2NZccc2",
        "c1ccNY[c:1]1X[c:3]2Sccc2",
        "c1cScY[c:1]1X[c:3]2cZScc2",
        "c1cScY[c:1]1X[c:3]2Sccc2",
        "c1ccS[c:1]1X[c:3]2Sccc2",
        # 5 member Ring with 6 member Ring
        "c1cNcY[c:1]1X[c:3]2cZcccc2",  # Pyrrol 1
        "c1ccNY[c:1]1X[c:3]2cZcccc2",  # Pyrrol 2
        "c1cccY[n:1]1X[c:3]2cZcccc2",  # Pyrrol 3
        "c1ccS[c:1]1X[c:3]2cZcccc2",  # Tiophen
        "c1cScY[c:1]1X[c:3]2cZcccc2",  # Tiophen 2
        "c1ccO[c:1]1X[c:3]2cZcccc2",  # Furan
        "c1cOcY[c:1]1X[c:3]2cZcccc2",  # Furan 2
        "c1cNnY[c:1]1X[c:3]2cZcccc2",  # Pyrazol 1
        "c1ccnY[n:1]1X[c:3]2cZcccc2",  # Pyrazol 2
        "n1cNcY[c:1]1X[c:3]2cZcccc2",  # Imidazol 1
        "n1ccNY[c:1]1X[c:3]2cZcccc2",  # Imidazol 1
        "c1nccY[n:1]1X[c:3]2cZcccc2",  # Imidazol 2
        "c1ncS[c:1]1X[c:3]2cZcccc2",  # Thiazol 1
        "n1ccS[c:1]1X[c:3]2cZcccc2",  # Thiazol 2
        "c1ncO[c:1]1X[c:3]2cZcccc2",  # Oxazol 1
        "n1ccO[c:1]1X[c:3]2cZcccc2",  # Oxazol 2
        "c1cOcY[c:1]1X[c:3]2cZccnc2",
        "c1ccO[c:1]1X[c:3]2cZccnc2",
        "c1ccO[c:1]1X[c:3]2cZcccn2",
        "c1cScY[c:1]1X[c:3]2cZccnc2",
        "c1ccS[c:1]1X[c:3]2cZccnc2",
        "c1ccS[c:1]1X[c:3]2cZcccn2",
        "c1cNcY[c:1]1X[c:3]2cZccnc2",
        "c1ccNY[c:1]1X[c:3]2cZccnc2",
        "c1ccNY[c:1]1X[c:3]2cZcccn2",
        "c1cNcY[c:1]1X[c:3]2nZccnc2",
        "c1ccNY[c:1]1X[c:3]2nZccnc2",
        # 6 member Rings with 6 member Rings
        # "C1CCCCN1X[c:3]2ccccc2", #Piperidine 1
        "c1cccnY[c:1]1X[c:3]2cZcccc2",  # Pyridin 2
        "c1cncnY[c:1]1X[c:3]2cZcccc2",  # Pyrimidin 1
        "n1cccnY[c:1]1X[c:3]2cZcccc2",  # Pyrimidin 2
        # double Ring with 6 member Rings
        "c13c(cccc3)ncY[n:1]1X[c:3]2cZcccc2",  # Benzimidazol
        "c13c(cccc3)ccY[n:1]1X[c:3]2cZcccc2",  # Indol 1
        "c13c(ccN3)cccY[c:1]1X[c:3]2cZcccc2",  # Indol 2
        "c13c(ncnc3)ncY[n:1]1X[c:3]2cZcccc2",  # Purin 1
        "n1c3c(ncnc3)nY[C:1]1X[c:3]2cZcccc2",  # Purin 2
    ]

    substituents = [
        "",
        "(C)",
        "(=O)",
        "(O)",
        "(Cl)",
        "(OC)",
        "(SC)",
        "(NC)",
        "(NC=O)",
        "(C4CC4)",
        "(C(C)C)",
        "(C(=O)N)",
        "(C(=O)O)",
        "(C(F)(F)F)",
    ]

    @classmethod
    def _subs_cleaning(cls, tmp_scaffolds:list[str], subs:str, reg_pattern:str):
        """

        Parameters
        ----------
        subs
        reg_pattern

        Returns
        -------

        """
        # Subst Prep - check that the correct atom label is there for marked atoms
        # -> otherwise radical!
        def pattern(x)->str:
            return re.findall(reg_pattern, x)[0]

        if subs == "":
            adapted = lambda x: (
                str(pattern(x))[:2] + "" + str(pattern(x))[2:]
                if (str(pattern(x))[1].islower())
                else str(pattern(x))[:2] + "H2" + str(pattern(x))[2:]
            )
            tmp_scaffolds = list(map(lambda x: x.replace(pattern(x), adapted(x)), tmp_scaffolds))
        else:
            adapted = lambda x: (
                str(pattern(x))[:2] + "" + str(pattern(x))[2:]
                if (str(pattern(x))[1].islower())
                else str(pattern(x))[:2] + "H" + str(pattern(x))[2:]
            )
            tmp_scaffolds = list(map(lambda x: x.replace(pattern(x), adapted(x)), tmp_scaffolds))

        return tmp_scaffolds

    @classmethod
    def _construct_dataset(
        cls, scaffolds: list[str], linkers: list[str], substituents:list[str]=None,
            linker_key:str="X"
    ) -> list[Chem.Mol]:
        """
        Generate the molecule data set

        Parameters
        ----------
        scaffolds: list[str]
            ring scaffolds for the connection.
        linkers: list[str]
            connect two rings with linkers
        substituents: list[str]
            which substituents shall be added to the rings
        linker_key: str
            str motive for linking

        Returns
        -------
        list[Chem.Mol]
            generated molecules
        """
        if substituents is None:
            smiles = []
            for linker in linkers:
                pattern = lambda x: re.findall(r"\[\w+:3\]", x)[0]
                if ":4]" in linker:
                    tmp_scaffolds = list(
                        map(
                            lambda x: x.replace(
                                str(pattern(x)), str(pattern(x)).replace("[", "").replace(":3]", "")
                            ),
                            scaffolds,
                        )
                    )
                elif ":3]" in linker:
                    tmp_scaffolds = list(map(lambda x: x.replace(":3]", ":4]"), scaffolds))
                elif ":2]" not in linker:
                    tmp_scaffolds = list(map(lambda x: x.replace(":1]", ":2]"), scaffolds))
                else:
                    tmp_scaffolds = scaffolds

                smile = [s.replace(linker_key, linker) for s in tmp_scaffolds]
                smiles.extend(smile)

            mols = [Chem.MolFromSmiles(m) for m in smiles]
            return mols
        else:
            def pattern(x):
                return re.findall(r"\[\w:3\]", x)[0]

            smiles = []
            for linker in linkers:
                for i, subs_y in enumerate(substituents):
                    for subs_z in substituents[i:]:
                        tmp_scaffolds = scaffolds

                        # Subs prep
                        tmp_scaffolds = cls._subs_cleaning(
                            tmp_scaffolds=tmp_scaffolds, subs=subs_y, reg_pattern=r"\[\w:1\]"
                        )
                        tmp_scaffolds = cls._subs_cleaning(
                            tmp_scaffolds=tmp_scaffolds, subs=subs_z, reg_pattern=r"\[\w:3\]"
                        )

                        # Linker Prep

                        if ":4]" in linker:
                            tmp_scaffolds = list(
                                map(
                                    lambda x: x.replace(str(pattern(x)), str(pattern(x))[1]),
                                    scaffolds,
                                )
                            )
                        if ":3]" in linker:
                            tmp_scaffolds = list(
                                map(lambda x: x.replace(":3]", ":4]"), tmp_scaffolds)
                            )
                        elif ":2]" not in linker:
                            tmp_scaffolds = list(
                                map(lambda x: x.replace(":1]", ":2]"), tmp_scaffolds)
                            )

                        # print(subs_y, subs_z, tmp_scaffolds)
                        smile = [
                            s.replace(linker_key, linker).replace("Y", subs_y).replace("Z", subs_z)
                            for s in tmp_scaffolds
                        ]
                        smiles.extend(smile)

        smiles = list(sorted(set(smiles)))
        # translate to Mol:
        mols = []
        fail_mol = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                fail_mol.append(smi)
                # print(smi)
            else:
                # print(smi)
                mol = Chem.AddHs(mol)
                torsion_atom_ids = cls._get_tors_atom_ids(mol)

                # print(i, real_torsion_atom_ids)
                mol.SetProp("torsion_atom_ids", str(torsion_atom_ids))
                mols.append(mol)

        print("Generated:", len(mols))
        print("Failed Smiles: ", len(fail_mol))
        return mols

    @staticmethod
    def _add_atom_to_incomplete_torsion(mol: Chem.Mol, torsion_id_maps:tuple[int, int, int, int],
                                        target_id:int) -> int :
        """
            add an atom to build a torsion, that was not selected

        Parameters
        ----------
        mol: Chem.Mol
            molecule where a torsion is build in.
        torsion_id_maps:

        target_id: int
            add a connected to targetID atom

        Returns
        -------
        int
            found atom

        """
        last_atom_id = target_id
        last_atom = mol.GetAtomWithIdx(last_atom_id)

        bound_atoms = []
        bonds = last_atom.GetBonds()
        for b in bonds:
            if b.GetBeginAtomIdx() != last_atom_id:
                bound_atoms.append(b.GetBeginAtomIdx())
            else:
                bound_atoms.append(b.GetEndAtomIdx())

        filtered_bound_atoms = list(filter(lambda x: x not in torsion_id_maps, bound_atoms))
        new_atom = mol.GetAtomWithIdx(filtered_bound_atoms[0])
        return new_atom

    @classmethod
    def _get_tors_atom_ids(cls, mol: Chem.Mol) -> list[int]:
        """
            get torsion atoms

        Parameters
        ----------
        mol: Chem.Mol

        Returns
        -------
        list[int]
            atom indices
        """
        atoms = {}
        for a in mol.GetAtoms():
            if a.GetAtomMapNum() > 0:
                atoms[a.GetAtomMapNum()] = a.GetIdx()

        if len(atoms) == 2:
            additional_atom = cls._add_atom_to_incomplete_torsion(
                mol, torsion_id_maps=list(atoms.values()), target_id=atoms[2]
            )
            additional_atom.SetAtomMapNum(1)
            atoms[1] = additional_atom.GetIdx()

            additional_atom = cls._add_atom_to_incomplete_torsion(
                mol, torsion_id_maps=list(atoms.values()), target_id=atoms[3]
            )
            additional_atom.SetAtomMapNum(4)
            atoms[4] = additional_atom.GetIdx()

        elif len(atoms) == 3:
            additional_atom = cls._add_atom_to_incomplete_torsion(
                mol, torsion_id_maps=list(atoms.values()), target_id=atoms[3]
            )
            additional_atom.SetAtomMapNum(4)
            atoms[4] = additional_atom.GetIdx()

        assert len(list(atoms.values())) == 4
        atoms_ids = list(map(lambda y: y[1], sorted(atoms.items(), key=lambda x: x[0])))
        return atoms_ids

    @classmethod
    def _create_sdf(cls, scaffolds: list[str], linkers: list[str], out_sdf_name: str,
                    substituents: list[str]=None) -> list[Chem.Mol]:
        """
            create sdf with all constructed data.

        Parameters
        ----------
        scaffolds: list[str]
        linkers: list[str]
        out_sdf_name: str
        substituents: list[str]

        Returns
        -------
        list[Chem.Mol]
            constructed molecules
        """

        mols = cls._construct_dataset(
            scaffolds=scaffolds, linkers=linkers, substituents=substituents
        )

        for m in mols:
            Chem.rdDistGeom.EmbedMolecule(m)

        writer = Chem.SDWriter(out_sdf_name)
        for mol in mols:
            writer.write(mol)
        writer.close()

        return mols

    @staticmethod
    def _get_data(sdf_path: str) -> list[Chem.Mol]:
        """
            read in a constructed database

        Parameters
        ----------
        sdf_path: str
            with molecule db
        Returns
        -------
        list[Chem.Mol]
            read in data
        """
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, strictParsing=True)
        mols = list(suppl)

        return mols

    @classmethod
    def get_carbon_scaffolds(cls) -> list[Chem.Mol]:
        """
        build or load: generate all carbon combinations

        Returns
        -------
        list[Chem.Mol]
            list of generated molecules

        """
        file_name = cls.out_folder_prefix + "/systematic_linked_carbon_rings.sdf"

        if bash.path.exists(file_name):
            mols = cls._get_data(file_name)
        else:
            mols = cls._create_sdf(
                scaffolds=cls.carbon_scaffolds, linkers=cls.linkers, out_sdf_name=file_name
            )
        return mols

    @classmethod
    def get_heterocycle_scaffolds(cls) -> list[Chem.Mol]:
        """
        build or load: generate combinations with hetero cycles

        Returns
        -------
        list[Chem.Mol]
            list of generated molecules

        """
        file_name = cls.out_folder_prefix + "/systematic_linked_hetero_rings.sdf"

        if bash.path.exists(file_name):
            mols = cls._get_data(file_name)
        else:
            mols = cls._create_sdf(
                scaffolds=cls.heteroCycle_scaffolds, linkers=cls.linkers, out_sdf_name=file_name
            )

        return mols

    @classmethod
    def get_scaffolds(cls):
        """
        build or load: return all scaffold structures

        Returns
        -------
        list[Chem.Mol]
            list of generated molecules
        """
        file_name = cls.out_folder_prefix + "/systematic_linked_rings.sdf"
        smiles_templates = cls.carbon_scaffolds + cls.heteroCycle_scaffolds

        if bash.path.exists(file_name):
            mols = cls._get_data(file_name)
        else:
            mols = cls._create_sdf(
                scaffolds=smiles_templates, linkers=cls.linkers, out_sdf_name=file_name
            )

        return mols

    @classmethod
    def get_carbon_substituent_scaffolds(cls):
        """
        build or load: add substitutents all carbon molecule combinations

        Returns
        -------
        list[Chem.Mol]
            list of generated molecules
        """
        file_name = cls.out_folder_prefix + "/systematic_linked_ortho_substituted_carbon_rings.sdf"

        if bash.path.exists(file_name):
            mols = cls._get_data(file_name)
        else:
            mols = cls._create_sdf(
                scaffolds=cls.carbon_scaffolds_withSubst,
                linkers=cls.linkers,
                substituents=cls.substituents,
                out_sdf_name=file_name,
            )

        return mols

    @classmethod
    def get_heterocycle_substituent_scaffolds(cls):
        """
        build or load: add substituents to hetero cycle molecule combinations

        Returns
        -------
        list[Chem.Mol]
            list of generated molecules
        """
        file_name = cls.out_folder_prefix + "/systematic_linked_ortho_substituted_hetero_rings.sdf"

        if bash.path.exists(file_name):
            mols = cls._get_data(file_name)
        else:
            mols = cls._create_sdf(
                scaffolds=cls.heteroCycle_Subst_scaffolds,
                linkers=cls.linkers,
                substituents=cls.substituents,
                out_sdf_name=file_name,
            )

        return mols

    @classmethod
    def get_substituent_scaffolds(cls, do_reconstruct:bool=False) -> list[Chem.Mol]:
        """
        build or load: scaffold mols with added substitutents

        Parameters
        ----------
        do_reconstruct: bool
            force reconstruction.

        Returns
        -------
        list[Chem.Mol]
            list of generated molecules
        """
        file_name = cls.out_folder_prefix + "/systematic_linked_ortho_substituted_rings.sdf"

        smiles_templates = cls.carbon_scaffolds_withSubst + cls.heteroCycle_Subst_scaffolds

        if bash.path.exists(file_name):
            mols = cls._get_data(file_name)
        elif not bash.path.exists(file_name) or do_reconstruct:
            mols = cls._create_sdf(
                scaffolds=smiles_templates,
                linkers=cls.linkers,
                substituents=cls.substituents,
                out_sdf_name=file_name,
            )

        return mols


if __name__ == "__main__":
    SystematicSet.get_carbon_scaffolds()
    SystematicSet.get_heterocycle_scaffolds()
    SystematicSet.get_scaffolds()
    SystematicSet.get_carbon_substituent_scaffolds()
    SystematicSet.get_heterocycle_substituent_scaffolds()
    SystematicSet.get_substituent_scaffolds()
