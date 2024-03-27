from pomdp_py.framework.generalization import Vector, RewardCost

description = "testing framework generalization"


def test_assign():
    v = Vector()
    assert v == [0.]

    v = Vector((2, 4, 8))
    assert v == [2., 4., 8.]

    v = Vector()
    assert v != [1.]


def test_add():
    r = RewardCost(0., Vector([0., 10.])) + RewardCost(10., Vector([90., 13.]))
    assert r.reward == 10.
    assert r.cost == [90., 23.]
    

def run():
    test_assign()
    test_add()


if __name__ == "__main__":
    run()