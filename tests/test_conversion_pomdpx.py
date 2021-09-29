# Tests the conversion to .pomdpx file format code

import sys
from pomdp_py.utils.test_utils import *
from pomdp_py.utils.interfaces.conversion import to_pomdpx_file
import os

description="testing conversion to .pomdpx file"

def test_pomdpx_file_conversion(pomdpconvert_path):
    """
    test pomdpx file conversion.
    If pomdp file conversion works, and .pomdpx file is created,
    then we assume the .pomdpx file is correct (because it is
    converted using an external software).
    """
    print("[testing] test_pomdpx_file_conversion")
    tiger = make_tiger()

    filename = "./test_tiger.POMDPX"
    print("[testing] converting to .pomdpx file")
    to_pomdpx_file(tiger.agent, pomdpconvert_path,
                   output_path=filename,
                   discount_factor=0.95)
    assert os.path.exists(filename), ".pomdpx file not created."
    print("Pass.")

    # Remove file
    os.remove(filename)

def _check_pomdpconvert():
    pomdpconvert_path = os.getenv("POMDPCONVERT_PATH")
    if pomdpconvert_path is None or not os.path.exists(pomdpconvert_path):
        raise FileNotFoundError("To run this test, download sarsop from "
                                "https://github.com/AdaCompNUS/sarsop. Then, follow the "
                                "instructions on this web page to compile this software. "
                                "Finally, set the environment variable POMDPCONVERT_PATH "
                                "to be the path to the pomdpconvert binary file "
                                "generated after compilation, likely located at "
                                "/path/to/sarsop/src/pomdpconvert")
    return pomdpconvert_path

def run():
    pomdpconvert_path = _check_pomdpconvert()
    test_pomdpx_file_conversion(pomdpconvert_path)


if __name__ == "__main__":
    run()
