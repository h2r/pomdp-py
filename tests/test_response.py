from pomdp_py.framework.basics import Response

description = "testing framework basics response"


def test_assign():
    r = Response()
    assert r.reward == 0.0
    
    r = Response(34.0)
    assert r.reward == 34.0
    
    
def test_add():
    r = Response()
    r = r + Response(42.0)
    assert r.reward == 42.0
    
    r = Response()
    r = r + 61.0
    assert r.reward == 61.0
    

def test_multiply():
    r = Response(1.0)
    r = r * 1000.0
    assert r.reward == 1000.0
    
    
def run():
    test_assign()
    test_add()
    test_multiply()
    

if __name__ == "__main__":
    run()
    