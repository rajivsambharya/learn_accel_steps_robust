import pytest
import numpy as np
import jax
import jax.numpy as jnp

from lah.pep import *

# Disable JIT for debugging
jax.config.update("jax_disable_jit", True)

# @pytest.fixture
# def vanilla_setup():
#     mu = 1
#     L = 10
#     num_iter = 10
#     # inv_cond = mu / L

#     # Default theoretical parameters for Nesterov (strongly convex)
#     alpha = jnp.repeat(1 / L, num_iter)
#     beta = jnp.array(np.random.rand(num_iter)) # a random beta vector where the beta values are uniformly drawn from [0,1]

#     A = build_A_matrix_with_xstar(alpha, beta)
#     return mu, L, num_iter, A, alpha, beta

# @pytest.fixture
# def proxgd_setup():
#     mu = 1
#     L = 10
#     num_iter = 25
#     # inv_cond = mu / L

#     # Default theoretical parameters for Nesterov (strongly convex)
#     alpha = jnp.repeat(1 / L, num_iter)
#     beta = jnp.array(np.random.rand(num_iter)) # a random beta vector where the beta values are uniformly drawn from [0,1]

#     A = build_A_matrix_prox_with_xstar(alpha, beta)
#     return mu, L, num_iter, A, alpha, beta

# def test_smooth_nesterov_pep_layer_matches_pepit(vanilla_setup):
#     mu, L, num_iter, A, alpha, beta = vanilla_setup

#     # Run our SDP layer
#     layer_nesterov = create_nesterov_pep_sdp_layer(L, num_iter)
#     G, F = layer_nesterov(A, solver_args={"verbose": True})
#     our_tau = float(F[-2] - F[-1])

#     # Run baseline from pepit
#     pepit_tau = float(pepit_nesterov(0, L, np.column_stack((alpha, beta))))

#     print("Our tau:", our_tau)
#     print("PEPit tau:", pepit_tau)

#     # Assert that the two values are close
#     # np.testing.assert_allclose(our_tau, pepit_tau, rtol=1e-3, atol=1e-6)
#     # np.testing.assert_allclose(our_tau, pepit_tau, rtol=1e-4, atol=1e-6)
#     np.testing.assert_allclose(our_tau, pepit_tau, atol=1e-4)

# def test_smooth_nesterovprox_pep_layer_matches_pepit(proxgd_setup):
#     mu, L, num_iter, A, alpha, beta = proxgd_setup

#     layer_nesterovprox = create_proxgd_pep_sdp_layer(L, num_iter)
#     G, F, H = layer_nesterovprox(A, solver_args={"verbose": True})
#     our_tau = float(F[-2] + H[-2] - F[-1] - H[-1])

#     pepit_tau = float(pepit_accel_gd(0, L, np.column_stack((alpha, beta)), False, True, "func"))

#     print("Our tau:", our_tau)
#     print("PEPit tau:", pepit_tau)

#     # Assert that the two values are close
#     # np.testing.assert_allclose(our_tau, pepit_tau, rtol=1e-3, atol=1e-6)
#     # np.testing.assert_allclose(our_tau, pepit_tau, rtol=1e-4, atol=1e-6)
#     np.testing.assert_allclose(our_tau, pepit_tau, atol=1e-4)

# def test_strcvx_smooth_quad_pep_layer_matches_pepit(vanilla_setup):
#     mu, L, num_iter, A, alpha, beta = vanilla_setup

#     layer_quad = create_quad_pep_sdp_layer(mu, L, num_iter)
#     G = layer_quad(A, solver_args={"eps": 1e-5, "verbose": True})
#     our_tau = float(G[0][2*num_iter + 2, 2*num_iter + 2]) ** .5

#     pepit_tau = float(pepit_accel_gd(mu, L, np.column_stack((alpha, beta)), True, False, "dist"))

#     print("Our tau:", our_tau)
#     print("PEPit tau:", pepit_tau)

#     np.testing.assert_allclose(our_tau, pepit_tau, rtol = 1e-4, atol = 1e-6)

# def test_strcvx_smooth_quadprox_pep_layer_matches_pepit(proxgd_setup):
#     mu, L, num_iter, A, alpha, beta = proxgd_setup

#     layer_quadprox = create_quadprox_pep_sdp_layer(mu, L, num_iter)
#     G, H = layer_quadprox(A, solver_args={"eps": 1e-5, "verbose": True})
#     our_tau = float(G[3*num_iter + 7, 3*num_iter + 7] - 2 * G[3*num_iter + 7, 1] + G[1, 1]) ** .5

#     pepit_tau = float(pepit_accel_gd(mu, L, np.column_stack((alpha, beta)), True, True, "dist"))

#     print("Our tau:", our_tau)
#     print("PEPit tau:", pepit_tau)

#     # np.testing.assert_allclose(our_tau, pepit_tau, rtol = 1e-2, atol = 1e-4)
#     np.testing.assert_allclose(our_tau, pepit_tau, atol = 1e-4)

# def test_quad():
#     alpha = np.array([])
#     ,0,1,2,3,4
#     0,0.0024845101625085367,0.002535515366784915,2.4760344871520004,0.03372036831533096,9.335166029096564
#     1,1.0,1.0,0.2401589860459823,4.694751043000144,1.0
#     2,1.0,1.0,1.668483497296194,0.00935863635156943,1.0
#     3,1.0,1.0,1.3291850953535689,0.031834606730397515,1.0
#     4,1.0,1.0,0.9585571811215076,0.0363699891260938,1.0
#     5,1.0,1.0,1.4078120355535593,0.03363731081113886,1.0
#     6,1.0,1.0,1.126108438178344,0.06592522753315531,1.0
#     7,1.0,1.0,1.5487698001239962,0.05360163166529464,1.0
#     8,1.0,1.0,1.438392120579283,0.057387448461689805,1.0
#     9,1.0,1.0,1.5724683146509295,0.05842786110942646,1.0
#     10,1.0,1.0,1.6518558452139034,0.05577980453418948,1.0
#     11,1.0,1.0,1.6828114943973007,0.053302339794045216,1.0
#     12,1.0,1.0,1.555676423038449,0.057179381678967134,1.0
#     13,1.0,1.0,1.5404476933150155,0.04855012225625057,1.0
#     14,1.0,1.0,1.2289581135185663,0.07579687440027573,1.0
#     15,1.0,1.0,1.4660293652230847,0.06072627139382027,1.0
#     16,1.0,1.0,1.523647075889661,0.05931771091208605,1.0
#     17,1.0,1.0,1.4795989225931854,0.05149541423658293,1.0
#     18,1.0,1.0,1.0393663524080938,0.04898747272678306,1.0
#     19,1.0,1.0,1.0,0.1353352832366127,1.0
#     20,1.0,1.0,1.0,2.061153622438558e-09,1.0

    
def test_fixed_point():
    num_iters = 20
    alpha = 2. * np.ones(num_iters)
    # alpha[2:10] = 1.5
    # alpha[0] = 2.5
    beta = -0.0 * np.ones(num_iters)
    # beta[1] = -4.7
    # beta[2] = -.05
    params = np.column_stack((alpha, beta))
    out = pepit_fixed_point_anderson(params)
    # out = pepit_fixed_point(params)
    import pdb
    pdb.set_trace()