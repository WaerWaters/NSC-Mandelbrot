"""
Test suite for Mandelbrot set implementations (M1: Test Suite 44 & 45).

Structure:
  1. Analytical Points    - Parametrized tests with mathematical justifications.
  2. Worker Isolation     - Testing the pure mandelbrot_pixel function.
  3. Cross-Validation     - Naive loop oracle vs Serial/Multiprocessing (32x32).
  4. Float32 Validation   - Comparing float32 versions against each other.
  5. Dask Integration     - Testing submit/gather on the compute function.
  6. Performance          - Relative regression (Numba vs Naive).
"""

import numpy as np
import pytest
import time
from parallel_mandelbrot import mandelbrot_pixel, mandelbrot_serial, mandelbrot_chunk, mandelbrot_parallel
from mandelbrotFunc import compute_mandelbrot_naive
from mini3Code import escape_count
from dask.distributed import Client, LocalCluster

# --- Helper: Python version for Numba comparison ---
# Requirement: "The compiled function is behaviourally identical to the Python version"
def mandelbrot_pixel_python(c_real, c_imag, max_iter):
    """Reference Python implementation matching Numba's check-before-update logic."""
    z_real = z_imag = 0.0
    for i in range(max_iter):
        if z_real*z_real + z_imag*z_imag > 4.0:
            return i
        z_real_next = z_real*z_real - z_imag*z_imag + c_real
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = z_real_next
    return max_iter

# --- 1. Analytical Point Tests ---
# Requirement: "Analytically provable - state why."
# Requirement: "Test the Python version first; the Numba version passes same tests."
@pytest.mark.parametrize("c, expected, reason", [
    (0j, 100, "Origin: z stays at 0 forever, bounded."),
    (-1+0j, 100, "Period-2 orbit: 0 -> -1 -> 0 -> -1, bounded."),
    (1+0j, 3, "Escapes: 0 -> 1 -> 2 -> 5. |5| > 2 at iteration 3."),
    (0.25+0j, 100, "Main Cardioid Cusp: converges to 0.5, bounded."),
], ids=["origin", "period-2", "escape", "cardioid-cusp"])
def test_analytical_points(c, expected, reason):
    """Verify correctness on points with mathematically proven behaviour."""
    # Test Python version first
    py_res = mandelbrot_pixel_python(c.real, c.imag, 100)
    assert py_res == expected, f"Python failed for {c}: {reason}"
    
    # Numba version passes the same tests
    nb_res = mandelbrot_pixel(c.real, c.imag, 100)
    assert nb_res == expected, f"Numba failed for {c}: {reason}"

# --- 2. Worker Isolation ---
# Requirement: "Test the worker function in isolation - it is pure."
def test_worker_isolation():
    """Test mandelbrot_pixel in isolation with clear inside/outside points."""
    assert mandelbrot_pixel(2.0, 2.0, 100) < 100, "Point (2,2) must escape."
    assert mandelbrot_pixel(0.0, 0.0, 100) == 100, "Origin must be bounded."

# --- 3. Cross-Validation & Multiprocessing ---
# Requirement: "Cross-validation - naive loop as oracle (32x32)."
# Requirement: "Multiprocessing - integration test: assembled grid matches serial."
def test_cross_validation_and_mp():
    """Validate Serial and Multiprocessing against a pure Python oracle."""
    N, MAX_ITER = 32, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    
    # Pure Python Oracle
    dx = (X_MAX - X_MIN) / N
    dy = (Y_MAX - Y_MIN) / N
    oracle = np.empty((N, N), dtype=np.int32)
    for r in range(N):
        ci = Y_MIN + r * dy
        for col in range(N):
            oracle[r, col] = mandelbrot_pixel_python(X_MIN + col * dx, ci, MAX_ITER)
            
    # Test Numba Serial
    serial = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    np.testing.assert_array_equal(serial, oracle, err_msg="Serial grid != Oracle")
    
    # Test Multiprocessing integration
    parallel = mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER, n_workers=2)
    np.testing.assert_array_equal(parallel, oracle, err_msg="Parallel grid != Oracle")

# --- 4. Float32 Note ---
# Requirement: "Cross-validate float32 implementations against each other, not against float64."
def test_float32_cross_validation():
    """Verify float32 consistency between vectorised and naive implementations."""
    N, MAX_ITER = 32, 100
    x = np.linspace(-2.5, 1.0, N, dtype=np.float32)
    y = np.linspace(-1.25, 1.25, N, dtype=np.float32)
    C = (x[np.newaxis, :] + 1j*y[:, np.newaxis]).astype(np.complex64)
    
    # Implementation 1: Vectorised (NumPy-based)
    res_vec = escape_count(C, MAX_ITER)
    
    # Implementation 2: Naive loop with float32 precision
    def naive_f32(C_grid, m_iter):
        out = np.empty(C_grid.shape, dtype=np.int32)
        for r in range(C_grid.shape[0]):
            for c in range(C_grid.shape[1]):
                z = np.complex64(0)
                # escape_count in mini3Code uses check-after-update logic
                for i in range(m_iter):
                    z = z*z + C_grid[r, c]
                    if np.abs(z) > 2.0:
                        out[r, c] = i
                        break
                else:
                    out[r, c] = m_iter
        return out
        
    res_naive = naive_f32(C, MAX_ITER)
    np.testing.assert_array_equal(res_vec, res_naive, err_msg="Float32 versions disagree")

# --- 5. Dask Integration ---
# Requirement: "Integration: future = client.submit(f, arg); assert client.gather(future) == expected."
def test_dask_integration():
    """Test Dask integration using the client.submit/gather pattern."""
    # Test the underlying compute function, not the scheduler
    with LocalCluster(n_workers=1, threads_per_worker=1, dashboard_address=None) as cluster:
        with Client(cluster) as client:
            args = (0, 8, 8, -2.5, 1.0, -1.25, 1.25, 100)
            future = client.submit(mandelbrot_chunk, *args)
            result = client.gather(future)
            
            # Expected from serial execution of the same compute function
            expected = mandelbrot_chunk(*args)
            np.testing.assert_array_equal(result, expected)

# --- 6. Performance Regression ---
# Requirement: "Do not assert on absolute timing."
# Requirement: "Relative assertions: assert numpy time < 0.1 * naive time."
def test_performance_regression():
    """Verify that Numba provides at least a 10x speedup over naive Python."""
    N, MAX_ITER = 64, 50
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    
    # Requirement: "Warm-up call needed before timing."
    mandelbrot_serial(8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10)
    
    # Timing Naive implementation
    t0 = time.perf_counter()
    compute_mandelbrot_naive(X_MIN, X_MAX, Y_MIN, Y_MAX, N, N, MAX_ITER)
    t_naive = time.perf_counter() - t0
    
    # Timing Numba implementation
    t1 = time.perf_counter()
    mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    t_numba = t1 = time.perf_counter() - t1
    
    assert t_numba < 0.1 * t_naive, f"Numba ({t_numba:.4f}s) not 10x faster than Naive ({t_naive:.4f}s)"