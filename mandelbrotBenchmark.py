import time, statistics
import numpy

def benchmark (func, *args, n_runs=3):
    """ Time func , return median of n_runs . """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args )
        times.append(time.perf_counter() - t0)
        median_t = statistics.median(times)
        print(f"Median  {median_t:.4f}s"
        f"(min={min(times):.4f}, max={max(times):.4f})")
    return median_t, result

# time tests using row sums and column sums
N = 10000
A = numpy.random.rand(N, N)

# Quick: 0.1445s
def compute_row_sums(A):
    for i in range(N):
        s = numpy.sum(A[i, :])

# Slower than row sums: 0.4936s
def compute_column_sums(A):
    for j in range(N):
        s = numpy.sum(A[:, j])


A_f = numpy.asfortranarray(A)
# Slower than normal row-major: 0.5058s
def compute_asfortranarray_row_sums(A_f):
    for i in range(N):
        s = numpy.sum(A_f[i, :])

# Faster than norma column-major: 0.1402s
def compute_asfortranarray_column_sums(A_f):
    for j in range(N):
        s = numpy.sum(A_f[:, j])