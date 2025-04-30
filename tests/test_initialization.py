import os,sys
import pytest
import feelpp.core as fppc
import feelpp.toolboxes.core as tb
from feelpp_scimba.Poisson import Poisson, runLaplacianPk, runConvergenceAnalysis, plot_convergence, custom_cmap



@pytest.fixture(scope="module")
def feelpp_env():
    # Initialize Feel++ environment
    sys.argv = ["feelpp_app"]
    env = fppc.Environment(
        sys.argv,
        opts=tb.toolboxes_options("coefficient-form-pdes", "cfpdes"),
        config=fppc.localRepository('feelpp_cfpde')
    )
    return env
#
def test_environment_creation(feelpp_env):
    assert feelpp_env is not None
    # Optionally check that certain attributes exist
    assert hasattr(feelpp_env, 'worldComm')

def test_scimba_import(feelpp_env):
    try:
        from feelpp_scimba.Poisson import Poisson, runLaplacianPk, runConvergenceAnalysis, plot_convergence, custom_cmap
    except ImportError as e:
        pytest.fail(f"Failed to import feelpp_scimba: {e}")

@pytest.fixture(scope="module")
def P(feelpp_env):
    # Instantiate the Poisson solver
    return Poisson(dim=2)

def test_convergence_analysis_and_plot(P):
    import numpy as np
    # Analytical solution and RHS
    u_exact = '(1 - (x*x + y*y))'
    grad_u_exact = '{-2*x, -2*y}'
    rhs = '4'
    g = '0'

    # Mesh sizes
    hs = [0.1, 0.05, 0.025, 0.0125]
    measures = []

    # Collect error measures for each mesh size
    for h in hs:
        P(h=h, rhs=rhs, g=u_exact, plot=None, shape='Disk',
          u_exact=u_exact, grad_u_exact=grad_u_exact)
        assert hasattr(P, 'measures'), "Poisson.measures not set"
        measures.append(P.measures)


    assert len(measures) == len(hs)

    # Perform convergence analysis
    poisson_json = P.model
    df = runConvergenceAnalysis(P, json=poisson_json, measures=measures, dim=2, hs=hs, verbose=False)

    # skip the first row (NaN rates) and extract the two rate columns
    l2_rates = df["P1-poisson_L2-convergence-rate"].iloc[1:].to_numpy()
    h1_rates = df["P1-poisson_H1-convergence-rate"].iloc[1:].to_numpy()
    assert np.allclose(l2_rates, 2.0, rtol=1e-2), f"L2 rates not ≃2: {l2_rates}"
    assert np.allclose(h1_rates, 1.0, rtol=1e-2), f"H1 rates not ≃1: {h1_rates}"


