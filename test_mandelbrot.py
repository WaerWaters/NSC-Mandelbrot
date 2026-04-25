import numpy as np
import pytest

import MP3_imp.naive as naive
import MP3_imp.multiProcessing as multi_processing
import MP3_imp.dask_local as dask_local

RES = 1024
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25


# Test 1 — analytically provable single-point values
#    c=0+0i: z stays 0 forever, never escapes → MAX_ITER
#    c=-1+0i: z cycles 0 → -1 → 0 → ..., never escapes → MAX_ITER
#    c=10+0i: |z|² = 100 > 4 on the very first iteration → escapes at i=0
@pytest.mark.parametrize("x, y, expected", [
    (0.0, 0.0, 100),
    (-1.0, 0.0, 100),
    (10.0, 0.0, 0),
])
def test_analytical_points(x, y, expected):
    assert naive.get_pixel(x, y, 100) == expected


# Test 2 — cross-validation: naive oracle vs multiprocessing on a 32x32 grid

def test_cross_validation_naive_vs_multiprocessed():
    RES = 32
    naive_result = naive.create_mandelbrot(res=RES, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX)
    multiprocessing_result = multi_processing.create_mandelbrot(res=RES, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, workers=1, n_chunks=4)
    assert np.array_equal(naive_result, multiprocessing_result)



# ---------------------------------------------------------------------------
# Test 3 — dask compute function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x, y, expected", [
    (0.0, 0.0, 100),
    (-1.0, 0.0, 100),
    (10.0, 0.0, 0),
])
def test_dask_evaluate_point_matches_known_values(x, y, expected):
    assert dask_local.get_pixel(x=x, y=y, max_iter=100) == expected
