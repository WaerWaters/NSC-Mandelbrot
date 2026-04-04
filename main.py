import numpy
import cProfile, pstats
from mandelbrotFunc import compute_mandelbrot_naive, compute_mandelbrot_numpy, compute_mandelbrot_naive_numba, compute_mandelbrot_hybrid, compute_mandelbrot_numba_typed
from mandelbrotBenchmark import benchmark, bench
import time
import matplotlib.pyplot as plt

# -2, 1, -1.5, 1.5, 1024, 1024, 100, display=False
# t, M = benchmark(compute_asfortranarray_row_sums, A_f)

"""
Lecture 2, Milestone 4
256 = 0.0506s
512 = 0.3069s
1024 = 1.2093s
2048 = 4.6193s
4096 = 18.5580s
"""

# compute_mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100, display=False)

t, r = benchmark(compute_mandelbrot_naive_numba, -2.5, 1.0, -1.25, 1.25, 8192, 8192, 100)

print(f"Naive Numba: {t:.4f}s")
