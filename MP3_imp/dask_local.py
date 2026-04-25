import MP3_imp.multiProcessing as multi_processing
from dask import delayed
import dask
import numpy as np
from numba import njit

def create_mandelbrot(res, x_min, x_max, y_min, y_max, n_chunks):
    chunk_size = max(1, res // n_chunks)
    tasks = []
    row = 0
    while row < res:
        row_end = min(row + chunk_size, res)
        tasks.append(delayed(multi_processing.mandelbrot_chunk)(row, row_end, res, x_min, x_max, y_min, y_max))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)

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