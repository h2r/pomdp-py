from pomdp_py.framework.basics import Response, Vector

description = "testing framework basics response"


def test_assign():
    r = Response()
    assert r["reward"] == 0.0
    
    r = Response({"reward": 34.0, "cost": Vector([12.0, 53.0])})
    assert r["reward"] == 34.0
    assert r["cost"] == [12.0, 53.0]
    
    
def test_add():
    r = Response()
    r = r + Response({"reward": 42.0})
    assert r["reward"] == 42.0
    
    r = Response({"reward": 42.0, "cost": Vector([4.0, 9.0])})
    r = r + Response({"reward": 2.0, "cost": Vector([1.0, 2.0])})
    assert r["reward"] == 44.0
    assert r["cost"] == Vector([5.0, 11.0])
    

def test_multiply():
    r = Response({"reward": 1.0, "cost": Vector([3.5, 6.2, 9.1])})
    r = r * 1000.0
    assert r["reward"] == 1000.0
    assert r["cost"] == [3500.0, 6200.0, 9100.0]
    
    
def run():
    test_assign()
    test_add()
    test_multiply()
    

if __name__ == "__main__":
    run()
    