import pytest
import numpy as np
import jax
import jax.numpy as jnp

from lah.pep import (
    build_A_matrix_with_xstar,
    create_nesterov_pep_sdp_layer,
    pepit_nesterov
    )

# Disable JIT for debugging
jax.config.update("jax_disable_jit", True)

@pytest.fixture
def nesterov_setup():
    mu = 0
    L = 10
    num_iter = 10
    # inv_cond = mu / L

    # Default theoretical parameters for Nesterov (strongly convex)
    alpha = jnp.repeat(1 / L, num_iter)
    beta = jnp.array(np.random.rand(num_iter)) # a random beta vector where the beta values are uniformly drawn from [0,1]

    A = build_A_matrix_with_xstar(alpha, beta)
    return mu, L, num_iter, A, alpha, beta

def test_smooth_nesterov_pep_layer_matches_pepit(nesterov_setup):
    mu, L, num_iter, A, alpha, beta = nesterov_setup

    # Run our SDP layer
    layer_nesterov = create_nesterov_pep_sdp_layer(L, num_iter)
    G, F = layer_nesterov(A, solver_args={"solve_method": "CLARABEL", "verbose": True})
    our_tau = float(F[-2] - F[-1])
    
    layer_nesterov = create_nesterov_pep_sdp_layer(L, num_iter)
    G, F = layer_nesterov(A, solver_args={"solve_method": "CLARABEL", "verbose": True})
    our_tau2 = float(F[-2] - F[-1])

    # Run baseline from pepit
    pepit_tau = float(pepit_nesterov(0, L, np.column_stack((alpha, beta))))

    print("Our tau:", our_tau)
    print("Our tau2:", our_tau2)
    print("PEPit tau:", pepit_tau)

    # Assert that the two values are close
    # np.testing.assert_allclose(our_tau, pepit_tau, rtol=1e-3, atol=1e-6)
    np.isclose(our_tau, pepit_tau, rtol=1e-4, atol=1e-6)
    np.isclose(our_tau2, pepit_tau, rtol=1e-4, atol=1e-6)
    