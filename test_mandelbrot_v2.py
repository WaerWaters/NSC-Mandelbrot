import numpy as np
import pytest
import time

from parallel_mandelbrot import (
    mandelbrot_pixel,
    mandelbrot_chunk,
    mandelbrot_serial,
    mandelbrot_parallel
)
from mandelbrotFunc import compute_mandelbrot_naive

MAX_ITER = 100
X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
SMALL_N = 32

def _naive_oracle(N=SMALL_N, max_iter=MAX_ITER):
    dx = (X_MAX - X_MIN) / N
    dy = (Y_MAX - Y_MIN) / N
    grid = np.empty((N, N), dtype=np.int32)
    for r in range(N):
        ci = Y_MIN + r * dy
        for col in range(N):
            cr = X_MIN + col * dx
            zr, zi = 0.0, 0.0
            escaped = False
            for i in range(max_iter):
                if zr * zr + zi * zi > 4.0:
                    grid[r, col] = i
                    escaped = True
                    break
                zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
            if not escaped:
                grid[r, col] = max_iter
    return grid

# 1. Analytically provable points with parametrization
#    c=0+0i: z stays 0 forever, never escapes → MAX_ITER
#    c=-1+0i: z cycles 0 → -1 → 0 → ..., never escapes → MAX_ITER
#    c=10+0i: |z|² = 100 > 4 on the very first iteration → escapes at i=1
@pytest.mark.parametrize("cr, ci, expected", [
    (0.0,  0.0, MAX_ITER),
    (-1.0, 0.0, MAX_ITER),
    (10.0, 0.0, 1),
])
def test_analytical_points(cr, ci, expected):
    assert mandelbrot_pixel(cr, ci, MAX_ITER) == expected

# 2. Cross-validation: Numba serial vs Pure-Python oracle
def test_cross_validation_oracle():
    oracle = _naive_oracle()
    result = mandelbrot_serial(SMALL_N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    np.testing.assert_array_equal(result, oracle)

# 3. Worker isolation
def test_worker_isolation():
    assert mandelbrot_pixel(-0.5, 0.0, MAX_ITER) == MAX_ITER
    assert mandelbrot_pixel(3.0, 0.0, MAX_ITER) == 1

# 4. Multiprocessing Integration
def test_multiprocessing_matches_serial():
    serial = mandelbrot_serial(SMALL_N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
    parallel = mandelbrot_parallel(SMALL_N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER, n_workers=2)
    np.testing.assert_array_equal(parallel, serial)

# 5. Dask Integration
def test_dask_submit_gather():
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, dashboard_address=None)
    client = Client(cluster)
    try:
        future = client.submit(mandelbrot_chunk, 0, SMALL_N, SMALL_N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
        result = client.gather(future)
        oracle = _naive_oracle()
        np.testing.assert_array_equal(result, oracle)
    finally:
        client.close()
        cluster.close()

# 6. Performance regression
def test_performance_regression():
    N, mi = 64, 50

    # Warm-up: trigger Numba JIT compilation before timing
    mandelbrot_serial(8, X_MIN, X_MAX, Y_MIN, Y_MAX, mi)

    t0 = time.perf_counter()
    compute_mandelbrot_naive(X_MIN, X_MAX, Y_MIN, Y_MAX, N, N, mi)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, mi)
    t_numba = time.perf_counter() - t0

    # Relative assertion only — no absolute timing
    assert t_numba < 0.1 * t_naive