import pytest
import numpy as np
import jax
import jax.numpy as jnp

from lah.pep import *

# Disable JIT for debugging
jax.config.update("jax_disable_jit", True)

@pytest.fixture
def vanilla_setup():
    mu = 1
    L = 10
    num_iter = 10
    # inv_cond = mu / L

    # Default theoretical parameters for Nesterov (strongly convex)
    alpha = jnp.repeat(1 / L, num_iter)
    beta = jnp.array(np.random.rand(num_iter)) # a random beta vector where the beta values are uniformly drawn from [0,1]

    A = build_A_matrix_with_xstar(alpha, beta)
    return mu, L, num_iter, A, alpha, beta

@pytest.fixture
def proxgd_setup():
    mu = 1
    L = 10
    num_iter = 25
    # inv_cond = mu / L

    # Default theoretical parameters for Nesterov (strongly convex)
    alpha = jnp.repeat(1 / L, num_iter)
    beta = jnp.array(np.random.rand(num_iter)) # a random beta vector where the beta values are uniformly drawn from [0,1]

    A = build_A_matrix_prox_with_xstar(alpha, beta)
    return mu, L, num_iter, A, alpha, beta

def test_smooth_nesterov_pep_layer_matches_pepit(vanilla_setup):
    mu, L, num_iter, A, alpha, beta = vanilla_setup

    # Run our SDP layer
    layer_nesterov = create_nesterov_pep_sdp_layer(L, num_iter)
    G, F = layer_nesterov(A, solver_args={"verbose": True})
    our_tau = float(F[-2] - F[-1])

    # Run baseline from pepit
    pepit_tau = float(pepit_nesterov(0, L, np.column_stack((alpha, beta))))

    print("Our tau:", our_tau)
    print("PEPit tau:", pepit_tau)

    # Assert that the two values are close
    # np.testing.assert_allclose(our_tau, pepit_tau, rtol=1e-3, atol=1e-6)
    # np.testing.assert_allclose(our_tau, pepit_tau, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(our_tau, pepit_tau, atol=1e-4)

def test_smooth_nesterovprox_pep_layer_matches_pepit(proxgd_setup):
    mu, L, num_iter, A, alpha, beta = proxgd_setup

    layer_nesterovprox = create_proxgd_pep_sdp_layer(L, num_iter)
    G, F, H = layer_nesterovprox(A, solver_args={"verbose": True})
    our_tau = float(F[-2] + H[-2] - F[-1] - H[-1])

    pepit_tau = float(pepit_accel_gd(0, L, np.column_stack((alpha, beta)), False, True, "func"))

    print("Our tau:", our_tau)
    print("PEPit tau:", pepit_tau)

    # Assert that the two values are close
    # np.testing.assert_allclose(our_tau, pepit_tau, rtol=1e-3, atol=1e-6)
    # np.testing.assert_allclose(our_tau, pepit_tau, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(our_tau, pepit_tau, atol=1e-4)

def test_strcvx_smooth_quad_pep_layer_matches_pepit(vanilla_setup):
    mu, L, num_iter, A, alpha, beta = vanilla_setup

    layer_quad = create_quad_pep_sdp_layer(mu, L, num_iter)
    G = layer_quad(A, solver_args={"eps": 1e-5, "verbose": True})
    our_tau = float(G[0][2*num_iter + 2, 2*num_iter + 2]) ** .5

    pepit_tau = float(pepit_accel_gd(mu, L, np.column_stack((alpha, beta)), True, False, "dist"))

    print("Our tau:", our_tau)
    print("PEPit tau:", pepit_tau)

    np.testing.assert_allclose(our_tau, pepit_tau, rtol = 1e-4, atol = 1e-6)

def test_strcvx_smooth_quadprox_pep_layer_matches_pepit(proxgd_setup):
    mu, L, num_iter, A, alpha, beta = proxgd_setup

    layer_quadprox = create_quadprox_pep_sdp_layer(mu, L, num_iter)
    G, H = layer_quadprox(A, solver_args={"eps": 1e-5, "verbose": True})
    our_tau = float(G[3*num_iter + 7, 3*num_iter + 7] - 2 * G[3*num_iter + 7, 1] + G[1, 1]) ** .5

    pepit_tau = float(pepit_accel_gd(mu, L, np.column_stack((alpha, beta)), True, True, "dist"))

    print("Our tau:", our_tau)
    print("PEPit tau:", pepit_tau)

    # np.testing.assert_allclose(our_tau, pepit_tau, rtol = 1e-2, atol = 1e-4)
    np.testing.assert_allclose(our_tau, pepit_tau, atol = 1e-4)