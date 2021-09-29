# Run all tests

import os
import sys
import argparse
import importlib
import time
from pomdp_py.utils import typ

ABS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_DIR)

def main():
    parser = argparse.ArgumentParser(description="Running tests.")
    args = parser.parse_args()

    # load the test modules
    tests = []
    for fname in sorted(os.listdir(ABS_DIR)):
        if fname != "test_all.py" and fname.startswith("test") and fname.endswith(".py"):
            test_module = importlib.import_module(fname.split(".py")[0])
            tests.append(test_module)

    for i, test_module in enumerate(tests):
        print(typ.bold("[{}/{}] {}".format(i+1, len(tests), test_module.description)))

        old_stdout = sys.stdout
        try:
            test_module.run()
        except FileNotFoundError as ex:
            sys.stdout = old_stdout
            print("   Error:", str(ex))

if __name__ == "__main__":
    main()
