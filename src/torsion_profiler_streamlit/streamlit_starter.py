"""
This helper script is starting an interactive streamlit server for torsion_profiler! it
allows you to use openFF, Ani2X and XTB for Torsion profiling. Simply execute the command and
"""

import os
import argparse
import subprocess

path = os.path.dirname(__file__)


def call():
    """
    Start process for streamlit server
    """
    with subprocess.Popen([
        "streamlit",
        "run",
        os.path.join(path, "streamlit_torsionprofiler.py")]
    ) as process:
        process.wait()


def main():
    """
    Main hook, starting the streamlit server
    """
    prog = "torsion_profiler_stream"
    usage = """simply type the program name, and open the pasted link in your browser of choice."""
    description = """
        **Streamlit server for torsion_profiler.**
        
          this command will start a interactive streamlit server which allows you to have a nice 
          visualization for torsion_profiler. (experimental ;) ) just enter the Network adress 
          pasted in the sdtout into your browser and you can use openFF, Ani2X and XTB for Torsion 
          profiling in an interactive session.
        """
    epilog = """
    Looking forward to your feedback!"""

    parser = argparse.ArgumentParser(
        prog=prog, usage=usage, description=description, epilog=epilog, add_help=True
    )

    parser.parse_args()
    print(parser.description)
    print("\n\nWill start Streamlit-Server now:")
    call()


if __name__ == "__main__":
    main()
