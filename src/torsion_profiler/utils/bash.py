"""
This is a collection of important Bash commands.
"""

from shutil import copy, rmtree, move, which
from os import path, chdir, getcwd, chmod, chown, mkdir, popen, remove, makedirs, listdir, environ

import io
import stat
import time
import warnings
from glob import glob
import subprocess as sub
from typing import Union

chmod_755 = stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP + stat.S_IROTH + stat.S_IXOTH


def program_binary_found(binary_name: str) -> bool:
    return which(binary_name) is not None

def wait_for_file_system(
    check_paths: Union[str, list[str]],
    regex_mode: bool = False,
    max_waiting_iterations: int = 1000,
    verbose: bool = False,
    time_wait_s_for_filesystem=2,
) -> bool:
    """
    This function can be used to circumvent lsf lag times.
    Parameters
    ----------
    check_path: str - Path to file to check if existant
    max_waiting_iterations: int - maximal iteration Time
    verbose: bool - tell me if found
    Returns
    -------
    True
        on success
    """
    if isinstance(check_paths, str):
        check_paths = [check_paths]

    for check_path in check_paths:
        it = 0
        waiting = True
        while waiting and it < max_waiting_iterations:
            if regex_mode:
                waiting = len(glob(check_path)) <= 0
            else:
                waiting = not path.exists(check_path)
            time.sleep(time_wait_s_for_filesystem)
            it += 1
        if waiting:
            raise IOError("Could not find file: " + check_path)
        if verbose:
            print("File Check FOUND: \t", check_path)

    return True


def check_path_dependencies(
    check_required_paths: Union[dict[any, str], list[str]],
    check_warn_paths: Union[str, list[str]] = None,
    verbose: bool = True,
) -> str:
    """check_path_dependencies
        checks a list of dependencies if each path is present or not. throws an
        IOError, if an Path is not existing.
    Parameters
    ----------
    check_required_paths :    Union[t.dict[any, str], list[str]]
        if path does not exist, an error wil be raised
    check_warn_paths:   Union[t.dict[any, str], list[str]]
        if path does not exist, a warning will be written out.
    verbose :   bool, optional
    Raises
    ------
    IOERROR
        if a file is not existing
    Returns
    -------
    Union[t.dict[any, str], list[str]]
        in_dependencies
    """
    if check_warn_paths is None:
        check_warn_paths = []

    found_error = False
    missing = []
    if verbose and isinstance(check_required_paths, list):
        print("\n\n==================\n\tCHECK dependencies\n")
        print("\n".join(list(map(lambda s: "Check " + str(s), check_required_paths))))
    elif verbose and isinstance(check_required_paths, dict):
        print("\nCHECK dependencies")
        print(
            "\n".join(
                list(
                    map(
                        lambda s: "Check " + str(s),
                        [check_required_paths[x] for x in check_required_paths],
                    )
                )
            )
        )

    # ERROR
    if isinstance(check_required_paths, dict):
        for x in check_required_paths:
            if "*" in x or "?" in x:
                if verbose:
                    print("Skipping regex")
                continue
            if verbose:
                print(x)
            if not isinstance(check_warn_paths[x], str) or (
                isinstance(check_required_paths[x], str)
                and not path.exists(check_required_paths[x])
            ):
                found_error = True
                missing.append(x)
    elif isinstance(check_required_paths, list):
        for x in check_required_paths:
            if "*" in x or "?" in x:
                if verbose:
                    print("Skipping regex")
                continue
            if verbose:
                print(x)
            if not isinstance(x, str) or (isinstance(x, str) and not path.exists(x)):
                found_error = True
                missing.append(x)

    # WARN
    if isinstance(check_required_paths, dict):
        for x in check_warn_paths:
            if verbose:
                print(x)
            if not isinstance(check_required_paths[x], str) or (
                isinstance(check_required_paths[x], str)
                and not path.exists(check_required_paths[x])
            ):
                warnings.warn(
                    "\tDid not find: " + str(x) + " with path: " + check_required_paths[x]
                )
    elif isinstance(check_warn_paths, list):
        for x in check_warn_paths:
            if verbose:
                print(x)
            if not isinstance(x, str) or (isinstance(x, str) and not path.exists(x)):
                warnings.warn("\tDid not find: " + str(x))

    if found_error:
        print("\n==================\nAUTSCH\n==================\n")
        missing_str = "\n\t".join(map(str, missing))
        raise IOError(
            "COULD NOT FIND all DEPENDENCY!\n\t Could not find path to: \n\t" + str(missing_str),
            "\n\n",
        )
    if verbose:
        print("All dependencies are correct!", "\n\n")
    return 0


def execute_os(command: Union[str, list[str]], verbose: bool = False) -> io.FileIO:
    """execute
        DEAPRECIATED
        sadly I can only use popen, as otherwise euler refuses.
        this command executes a command on the os-layer. (e.g.: on linux a bash command)
        therefore it uses popen
    Parameters
    ----------
    command :   str, lists[str]
    verbose: bool, optional
    Raises
    ------
    OSError
        if bash command fails an OSError is raised
    Returns
    -------
    io.FileIO
        returns the execution log of the process.
    """

    class DummyProcess:
        """
        Dummy Process
        """
        def __init__(self, stdout, stderr, ret):
            self.stdout = stdout
            self.stderr = stderr
            self.poll = lambda x: int(ret)

    if isinstance(command, list):
        command = " ".join(command)

    if verbose:
        print(command + "\n")

    try:
        ret = popen(command)
        print(ret)
    except Exception as err:
        raise Exception(
            "could not execute bash command:\n  error: "
            + "\n\t".join(err.args)
            + "\n\t"
            + str(command)
            + "\n\tCommand returned: \t"
            + str(ret.read())
        )

    if verbose:
        print("\t" + "\n\t".join(ret.readlines()) + "\n")

    return DummyProcess(stdout=ret, stderr=[], ret=0)


def execute(
    command: Union[str, list[str]],
    catch_std: Union[bool, str] = False,
    env: dict = None,
    verbose: bool = False,
) -> sub.CompletedProcess:
    """execute_subprocess
        This command starts a subprocess, that is executing the str command in bash.
    Parameters
    ----------
    command : str
        bash command
    catch_std :
        if bool: catch the output and past it into the command line
        if str: catch output and write it into this file
    env: dict
        environment
    verbose : bool
        verbosity
    Returns
    -------
    CompletedProcess
        return the executed process obj. (from subprocess)
    """

    if isinstance(command, list):
        command = " ".join(command)
    if verbose:
        print("\texecute command: \n\t" + command)

    kwargs = {}
    if isinstance(catch_std, bool):
        kwargs.update({"stdout": sub.PIPE})
    elif isinstance(catch_std, str):
        kwargs.update({"stdout": open(catch_std, "w")})

    if env is None:
        env = environ.copy()

    p = sub.Popen(args=command, shell=True, stderr=sub.PIPE, env=env, **kwargs)

    try:
        p.wait(120)  # Wait for process to finish
    except sub.TimeoutExpired:
        print("\tWaiting ... ")
        p.wait()  # Wait for process to finish
    p.terminate()  # Make sure its terminated
    r = p.poll()

    if r:  # Did an Error occure?
        msg = "SubProcess Failed due to returncode: " + str(r) + "\n COMMAND: \n\t" + str(command)
        msg += "\nSTDOUT:\n\t"
        msg += "NONE" if (p.stdout is None) else "\n\t".join(map(str, p.stdout.readlines()))
        msg += "\nSTDERR:\n\t"
        msg += "NONE" if (p.stdout is None) else "\n\t".join(map(str, p.stderr.readlines()))
        raise ChildProcessError(msg)
    if verbose:
        print("RETURN: ", r)

    return p
