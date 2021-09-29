# Tests the conversion to .pomdp file format code

import os
import io
import sys
import pomdp_py
import subprocess
import inspect
from pomdp_py.utils.test_utils import *
from pomdp_py.utils.interfaces.conversion import to_pomdp_file


description="testing conversion to .pomdp file"

def test_pomdp_file_conversion(pomdp_solve_path):
    print("[testing] test_pomdp_file_conversion")
    tiger = make_tiger()

    # Generate a .pomdp file
    filename = "./test_tiger.POMDP"
    to_pomdp_file(tiger.agent, filename, discount_factor=0.95)
    assert os.path.exists(filename)

    print("[testing] Running pomdp-solve on generated %s" % filename)
    proc = subprocess.Popen([pomdp_solve_path, "-pomdp", filename],
                            stdout=subprocess.PIPE)
    solution_found = False
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        if "Solution found" in line:
            solution_found = True
    assert solution_found, "Something wrong - generated tiger POMDP could not be solved"
    print("Pass.")

    # Delete the generated pomdp file
    os.remove(filename)

    # Delete pomdp-solve generated files
    remove_files("./*.pg")
    remove_files("./*.alpha")

def _check_pomdp_solve():
    pomdp_solve_path = os.getenv("POMDP_SOLVE_PATH")
    if pomdp_solve_path is None or not os.path.exists(pomdp_solve_path):
        raise FileNotFoundError("To run this test, download pomdp-solve from "
                                "https://www.pomdp.org/code/. Then, follow the "
                                "instructions on this web page to compile this software. "
                                "Finally, set the environment variable POMDP_SOLVE_PATH "
                                "to be the path to the pomdp-solve binary file "
                                "generated after compilation, likely located at "
                                "/path/to/pomdp-solve-<version>/src/pomdp-solve ")
    return pomdp_solve_path

def run():
    pomdp_solve_path = _check_pomdp_solve()
    test_pomdp_file_conversion(pomdp_solve_path)

if __name__ == "__main__":
    run()
