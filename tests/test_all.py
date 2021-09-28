# Run all tests

import os
import sys
import argparse
import importlib

ABS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_DIR)

def main():
    parser = argparse.ArgumentParser(description="Running tests.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode; more things will be printed per test")
    args = parser.parse_args()

    # load the test modules
    tests = []
    for fname in sorted(os.listdir(ABS_DIR)):
        if fname in ["test_particles.py", "test_tree_debugger.py"]:#fname.startswith("test") and fname.endswith(".py"):
            test_module = importlib.import_module(fname.split(".py")[0])
            tests.append(test_module)

    for i, test_module in enumerate(tests):
        print("[{}/{}] {}".format(i+1, len(tests), test_module.description))

        if args.verbose:
            test_module.run()
        else:
            old_stdout = sys.stdout
            with open(os.devnull, 'w') as f:
                sys.stdout = f
                test_module.run()
            sys.stdout = old_stdout

if __name__ == "__main__":
    main()
