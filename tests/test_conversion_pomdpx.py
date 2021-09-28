# Tests the conversion to .pomdpx file format code

import sys
from pomdp_py.utils.test_utils import *
from pomdp_py.utils.interfaces.conversion import to_pomdpx_file
import os


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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("To run test, do: %s <pomdpconvert-path>" % sys.argv[0])
        print("Download pomdpconvert from https://github.com/AdaCompNUS/sarsop")
        exit(1)
    converter_path = sys.argv[1]
    test_pomdpx_file_conversion(converter_path)
