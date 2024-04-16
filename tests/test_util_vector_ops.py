from pomdp_py.utils.cvec import Vector


description = "testing utils cvec"


def test_assign():
    v = Vector([0])
    assert v == [0.]

    v = Vector([2, 4, 8])
    assert v == [2., 4., 8.]

    v = Vector([0])
    assert v != [1.]


def test_as_list():
    v = Vector([10., 3., 3.])
    assert v.as_list() == [10., 3., 3.]

    v = Vector([1., 5., 9., 11., 6.])
    assert v.as_list() == [1., 5., 9., 11., 6.]


def test_as_vector():
    v = Vector([1., 2., 3.])
    assert v.as_vector() == [1., 2., 3.]


def test_clip():
    v = Vector([2, 5, 7])
    assert Vector.clip(v, 0, 10) == [2., 5., 7.]

    v = Vector([2, 5, 7])
    assert Vector.clip(v, 0, 4) == [2., 4., 4.]

    v = Vector([2, 5, 7])
    assert Vector.clip(v, 4, 10) == [4., 5., 7.]

    v = Vector([2, 5, 7])
    assert Vector.clip(v, 3, 4) == [3., 4., 4.]


def test_copy():
    v = Vector([1., 2., 3.])
    assert v.copy() == [1., 2., 3.]


def test_dot():
    v0 = Vector([1., 3., 5., 7.])
    v1 = Vector([0., 13., 0., 10.])
    assert v0.dot(v1) == 109.


def test_fill():
    v0 = Vector.fill(10., 5)
    assert v0 == [10., 10., 10., 10., 10.]

    v1 = Vector.fill(3., 2)
    assert v1 == [3., 3.]


def test_len():
    v = Vector([1., 2.])
    assert v.len() == 2

    v = Vector([5., 7., 2.])
    assert v.len() == 3


def test_null():
    v = Vector.null(4)
    assert v == [0., 0., 0., 0.]


def test_get_and_set_item():
    v = Vector.null(3)
    v[0] = 1.
    v[2] = 1999.

    assert v == [1., 0., 1999.]
    assert v[0] == 1.
    assert v[1] == 0.
    assert v[2] == 1999.


def test_iter():
    v = Vector([1., 2., 4., 8.])
    for value0, value1 in zip(v, [1., 2., 4., 8.]):
        assert value0 == value1


def test_add():
    v0 = Vector([1, 2, 3])
    v1 = Vector([10, 22, 55])

    assert v0 + 4. == [5., 6., 7.]
    assert 4. + v0 == [5., 6., 7.]
    assert v0 + v1 == [11., 24., 58.]
    assert v1 + v0 == [11., 24., 58.]


def test_mul():
    v = Vector([9., 8.])
    assert v * 5. == [45., 40.]
    assert v * 10. == [90., 80.]


def test_sub():
    v0 = Vector([1, 2, 3])
    v1 = Vector([10, 22, 55])

    assert v0 - v1 == [-9., -20., -52.]
    assert v1 - v0 == [9., 20., 52.]
    assert v1 - 10. == [0., 12., 45.]
    assert v0 - 0. == [1., 2., 3.]


def test_truediv():
    v = Vector([10., 20., 50.])
    assert v / 2. == [5., 10., 25.]
    assert v / 20. == [0.5, 1.0, 2.5]


def test_str():
    v = Vector([2., 4.])
    assert str(v) == str([2., 4.])


def run():
    test_assign()
    test_as_list()
    test_as_vector()
    test_clip()
    test_copy()
    test_dot()
    test_fill()
    test_len()
    test_null()
    test_get_and_set_item()

    test_add()
    test_mul()
    test_sub()
    test_truediv()
    test_str()


if __name__ == "__main__":
    run()