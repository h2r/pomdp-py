import io
import os
import pomdp_py
import pickle
import subprocess
import textwrap

description="testing hashing & pickling some objects"

def test_hashing_pickling():
    objstate1 = pomdp_py.ObjectState("apple", {"color": "red", "weight": 1.5})
    objstate2 = pomdp_py.ObjectState("banana", {"color": "yellow", "weight": 0.5})

    assert hash(objstate1.copy()) == hash(objstate1)
    assert hash(objstate1.copy()) == hash(objstate1.copy())
    assert hash(objstate1.copy()) != hash(objstate2)
    assert hash(objstate1) != hash(objstate2.copy())
    assert hash(objstate2.copy()) == hash(objstate2)
    assert hash(objstate2.copy()) == hash(objstate2.copy())

    objstate3 = objstate1.copy()
    assert hash(objstate3) == hash(objstate1)
    objstate3["color"] = "green"
    # hashcode should be kept constant after the object's creation
    assert hash(objstate3) == hash(objstate1)
    assert objstate3 != objstate1

    oos1 = pomdp_py.OOState({1:objstate1, 2:objstate2})
    assert hash(oos1.copy()) == hash(oos1)

    # save oos1 as a pickle file
    temp_oos1_file = "temp-oos1.pkl"
    with open(temp_oos1_file, "wb") as f:
        pickle.dump(oos1, f)

    # Create this temporary program. Then run it with a different python process
    prog = """
    import pickle
    import pomdp_py
    with open('%s', "rb") as f:
        oos1_loaded = pickle.load(f)

    objstate1 = pomdp_py.ObjectState("apple", {"color": "red", "weight": 1.5})
    objstate2 = pomdp_py.ObjectState("banana", {"color": "yellow", "weight": 0.5})
    oos1 = pomdp_py.OOState({1:objstate1, 2:objstate2})
    assert hash(oos1) == hash(oos1_loaded), "hash code for object {} != hash code for pickled object {}".format(hash(oos1), hash(oos1_loaded))
    assert oos1 == oos1_loaded
    print("Passed.")
    """ % (temp_oos1_file)
    prog = textwrap.dedent(prog)
    temp_prog_file = "_temp_prog.py"
    with open(temp_prog_file, "w") as f:
        f.write(prog)

    proc = subprocess.Popen(["python", temp_prog_file],
                            stdout=subprocess.PIPE)
    passed = False
    try:
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
            if "Passed" in line:
                passed = True
                break
        assert passed, "Something wrong - pickled object does not equal to original object"
        print("Pass.")
    finally:
        os.remove(temp_oos1_file)
        os.remove(temp_prog_file)

def run():
    test_hashing_pickling()

if __name__ == "__main__":
    run()
