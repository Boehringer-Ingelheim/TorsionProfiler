"""
general utils
"""
import os
import inspect
import argparse
from types import FunctionType, MethodType

import pandas as pd
from rdkit import Chem
from . import bash
from . import store_mol_db


class RingError(Exception):
    """
    Special Exception for torsion found to be in ring.
    """


def str2bool(value: str) -> bool:
    """
        a simple fuction trying to translate a string to a bool.

    Parameters
    ----------
    v : str
        the potential bool

    Returns
    -------
    bool
        translated value

    Raises
    ------
    argparse.ArgumentTypeError
        if the value was not a bool.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def dynamic_parser(func: callable, title: str):
    """
        This function builds dynamically a parser obj for any function, that has parameters
        with annotated types. Result is beeing able to parse any function dynamically via bash.

    Parameters
    ----------
    func : callable
        the function that should be parsed
    title : str
        title for the parser

    Returns
    -------
    args
        The parsed arguments

    Raises
    ------
    IOError
        error if a parsing arg is unknown
    """
    parser = argparse.ArgumentParser(description=title)
    args = inspect.getfullargspec(func)
    total_defaults = len(args.defaults)
    total_args = len(args.args)
    total_required = total_args - total_defaults

    parser.description = func.__name__ + " - " + func.__doc__.split("Parameters")[0]
    for argument, argument_type in args.annotations.items():
        index = args.args.index(argument)
        required = True if index < total_required else False
        default = None if required else args.defaults[index - total_required]
        if argument_type is bool:
            argument_type = str2bool
        parser.add_argument("-" + argument, type=argument_type,
                            required=required, default=default)

    args, unkown_args = parser.parse_known_args()
    if len(unkown_args) > 0:
        raise IOError(__name__ + " got unexpected argument(s) for parser:\n" + str(unkown_args))
    return args


def dict_to_nice_string(control_dict: dict) -> str:
    """
        Converts a dictionary of options (like template_control_dict)
          to a more human readable format. Which can then be printed to a text file,
          which can be manually modified before submiting analysis jobs.

    Parameters
    ----------
    control_dict : dict
        analysis control dictonary

    Returns
    -------
    str
        nice formatting of the control dictionary for printing.
    """
    script_text = "control_dict = {\n"
    for key, value in control_dict.items():
        script_text += '\t"' + key + '": '
        first = False
        if isinstance(value, dict):
            if "do" in value:  # do should always be first in this list
                script_text += '{"do":' + str(value["do"]) + ","
                if len(value) > 1:
                    script_text += "\n"
                first = True
            for key2, value2 in value.items():  # alternative keys
                # prefix
                if first:
                    prefix = " "
                    first = False
                else:
                    prefix = "\t\t"

                # key_val
                if key2 == "do":
                    continue

                if isinstance(value2, dict):
                    script_text += (
                        prefix + '"' + str(key2) + '": ' + _inline_dict(value2,
                                                                        "\t\t\t") + ",\n"
                    )
                else:
                    script_text += prefix + '"' + str(key2) + '": ' + str(value2) + ","
            script_text += prefix + " },\n"
        else:
            script_text += str(value) + ",\n"
    script_text += "}\n"
    return script_text


def _inline_dict(in_dict: dict, prefix: str = "\t") -> str:
    """
        translate dictionary to one code line. can be used for meta-scripting

    Parameters
    ----------
    in_dict: dict
        analysis control dict
    prefix : str, optional
        prfix symbol to dict write out.

    Returns
    -------
    str
        code line.
    """
    msg = "{\n"
    for key, value in in_dict.items():
        if isinstance(value, dict):
            msg += (
                prefix
                + '"'
                + str(key)
                + '": '
                + _inline_dict(in_dict=value, prefix=prefix + "\t")
                + ","
            )
        else:
            msg += prefix + '"' + str(key) + '": ' + str(value) + ",\n"
    return msg + prefix + "}"


def write_job_script(
    out_script_path: str,
    target_function: callable,
    variable_dict: dict,
    out_rdkitMol_prefix: str = None,  # hm... ugly hack
    python_cmd: str = "python3",
    verbose: bool = False,
) -> str:
    """
        this function writes submission commands into a file. The command will be started
        from a bash env into python.

    Parameters
    ----------
    out_script_path: str
        path of the output script.
    target_function : callable
        the function, that shall be submitted
    variable_dict : dict
        variables for this function
    python_cmd : str, optional
        which python command shall be supplied
    verbose : bool, optional
        c'est la vie

    Returns
    -------
    str
        returns an out script path.

    Raises
    ------
    IOERROR
        if outpath is not possible
    ValueError
        if required variable from the var-dict for the function is missing
    """

    if not os.path.exists(os.path.dirname(out_script_path)):
        raise IOError(
            "Could not find path of dir, that should contain the schedule script!\n\t Got Path: "
            + out_script_path
        )

    # Build str:
    s = inspect.signature(target_function)  # to lazy for numpydoc
    import_string = ""
    import_string += "#IMPORTS\n"
    import_string += "from datetime import datetime\nstart=datetime.now()\n"
    vars_string = "#VARIABLES: \n"
    cmd_options = ""

    missed_keys = []
    for param_key in s.parameters:
        if param_key in variable_dict:
            dict_value = variable_dict[param_key]
            if isinstance(dict_value, dict):
                vars_string += dict_to_nice_string(dict_value)
            elif isinstance(dict_value, list):
                vars_string += param_key + "= [ " + ", ".join(map(str, dict_value)) + "]\n"
            elif isinstance(dict_value, str):
                vars_string += param_key + ' = "' + str(dict_value) + '"\n'
            elif isinstance(dict_value, pd.DataFrame) and "ROMol" in dict_value.columns:
                if ("torsion_profiler.utils import read_mol_db" not
                        in import_string):
                    import_string += ("from torsion_profiler.utils import read_mol_db\n")

                tmp_path = f"{os.path.dirname(out_script_path)}/in_{out_rdkitMol_prefix}_{param_key}.sdf"

                store_mol_db(df_mols=dict_value, out_sdf_path=tmp_path)

                vars_string += f'{param_key} = read_mol_db("{tmp_path}")\n'

            elif isinstance(dict_value, Chem.Mol):
                if "from rdkit import Chem" not in import_string:
                    import_string += "from rdkit import Chem\n"

                tmp_path = f"{os.path.dirname(out_script_path)}/in_{out_rdkitMol_prefix}_{param_key}.sdf"

                writer = Chem.SDWriter(tmp_path)
                out_mol = Chem.Mol(dict_value)
                for i in range(dict_value.GetNumConformers()):
                    conf_prop = dict_value.GetConformer().GetPropsAsDict()
                    mol_prop = dict_value.GetPropsAsDict()

                    for key, prop_value in mol_prop.items():
                        out_mol.SetProp(str(key), str(prop_value))

                    for key, prop_value in conf_prop.items():
                        out_mol.SetProp(str(key), str(prop_value))

                    if out_mol.HasProp("torsion_angles"):
                        out_mol.SetProp(
                            "torsion_angle", out_mol.GetProp("torsion_angles").split()[i]
                        )

                    out_mol.SetProp("conf_id", str(i))
                    writer.write(out_mol, i)
                writer.close()

                vars_string +=  (f'{param_key}s = [m for m in Chem.SDMolSupplier("{tmp_path}", '
                                 f'removeHs=False)]\n')

                #vars_string += key + " = " + key + "s[0]\n"
                vars_string += f"{param_key} = {param_key}s[0]\n"

                vars_string +=  (f"if(len({param_key}s)>1): [{param_key}.AddConformer("
                                 f"m.GetConformer(), i) for i,m in enumerate({param_key}s[1:], "
                                 f"start=1)]\n")
            else:
                vars_string += param_key + " = " + str(dict_value) + "\n"
            cmd_options += f"{param_key} = {param_key}, "
        elif s.parameters[param_key].default == inspect._empty:
            missed_keys.append(param_key)

    if len(missed_keys) > 0:
        raise ValueError(
            "Found some variables missing in variable dict,that are required!\n\t"
            + "\n\t".join(missed_keys)
        )

    cmd_string = "\n#DO\n"
    if isinstance(target_function, FunctionType):
        import_string += (
            "from " + str(target_function.__module__) + " import " + target_function.__name__
        )
        cmd_string += target_function.__name__ + "(" + cmd_options + ")"

    elif hasattr(target_function, "__self__") and isinstance(target_function, MethodType):
        obj = target_function.__self__
        object_name = "target_obj"
        class_str = obj._string_constructor(obj_name=object_name)

        # gen_nice formated script:
        lines = class_str.split("\n")
        for line in lines:
            if line.startswith("from") or line.startswith("import"):
                import_string += line + "\n"
            else:
                cmd_string += line + "\n"

        cmd_string += object_name + "." + target_function.__name__ + "(" + cmd_options + ")\n"
    else:
        raise ValueError("I did not understand the target! Got: " + str(target_function))

    timing_string = (
        'end=datetime.now()\nduration=end-start\nprint("Duration: ",duration.seconds, "s\\n")'
    )
    script_text = (
        "#!/usr/bin/env "
        + python_cmd
        + "\n\n"
        + import_string
        + "\n\n"
        + vars_string
        + "\n\n"
        + cmd_string
        + "\n"
        + timing_string
        + "\n\nexit(0)\n"
    )
    if verbose:
        print(script_text)

    # write out file
    with open(out_script_path, "w") as out_script_file:
        out_script_file.write(script_text)
        out_script_file.close()
        bash.chmod(out_script_path, bash.chmod_755)

    return out_script_path


