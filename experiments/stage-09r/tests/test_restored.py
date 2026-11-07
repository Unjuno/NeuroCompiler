import sys,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
def test_implementation_exists():
    assert importlib.util.find_spec('experiment') is not None
import json
import numpy as np
import pytest
@pytest.fixture(scope='module')
def check():
    import experiment as e
    e.verify()
    return json.loads((e.ROOT/'checks/numerical_tests.json').read_text())
def test_orthogonality(check): assert check['gram_max']<1e-12
def test_projection_decomposition(check): assert check['projection_identity_abs']<1e-11
def test_energy_derivative(check): assert check['energy_fd_abs']<1e-7
def test_hessian(check): assert check['hessian_fd_max']<1e-7 and check['hessian_min_eig']>0
def test_manufactured_solution(check): assert check['manufactured_max']<1e-10
def test_network_jvp(check): assert check['nn_jvp_max']<1e-12 and check['nn_forward_max']==0
def test_physics_jvp(check): assert check['physics_jvp_fd_max']<1e-6
def test_invalid_physics():
    import experiment as e
    with pytest.raises(ValueError): e.fields(np.array([.5,.5,-.1,10,.5,.5,.1,10,2,.2,1]),8)
