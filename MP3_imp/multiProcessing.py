from multiprocessing import Pool
import numpy as np
from numba import njit

def create_mandelbrot(res, x_min, x_max, y_min, y_max, workers, n_chunks):
    if n_chunks is None:
        n_chunks = workers
    chunk_size = max(1, res // n_chunks)
    chunks = []
    row = 0
    while row < res:
        row_end = min(row + chunk_size, res)
        chunks.append((row, row_end, res, x_min, x_max, y_min, y_max))
        row = row_end
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max)]
    with Pool(processes=workers) as pool:
        pool.map(_worker, tiny)
        parts = pool.map(_worker, chunks)
    return np.vstack(parts)

def _worker(args):
    return mandelbrot_chunk(*args)

@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, res, x_min, x_max, y_min, y_max):
    x = np.linspace(start=x_min, stop=x_max, num=res)
    y = np.linspace(start=y_min, stop=y_max, num=res)
    grid = np.empty((row_end - row_start, res), dtype=np.int32)
    for i in range(row_end - row_start):
        for j in range(res):
            n = get_pixel(x=x[row_start + i], y=y[j], max_iter=100)
            grid[i][j] = n
    return grid

@njit(cache=True)
def get_pixel(x, y, max_iter):
    c = complex(x, y)
    z = 0
    for i in range(max_iter):
        z = z*z + c
        if z.real*z.real + z.imag*z.imag > 4.0:
            return i
    return max_iter