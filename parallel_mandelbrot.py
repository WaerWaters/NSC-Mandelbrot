import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics, matplotlib.pyplot as plt
from pathlib import Path
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#M1

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0:
            return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)
    return out

def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#M2

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4, n_chunks=None, pool=None):
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_workers)
    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    
    if pool is not None:
        return np.vstack(pool.map(_worker, chunks))
    
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, tiny)
        parts = pool.map(_worker, chunks)
    
    return np.vstack(parts)

"""
if __name__ == '__main__':
    result = mandelbrot_parallel(1024, -2.5, 1.0, -1.25, 1.25, n_workers=4, n_chunks=4*4)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result, extent=[-2.5, 1.0, -1.25, 1.25], cmap='inferno', origin='lower', aspect='equal')
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    out = Path(__file__).parent / 'mandelbrot.png'
    fig.savefig(out, dpi=150)
    print(f'Saved: {out}')
"""

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#M3
"""
if __name__ == '__main__':
    N, max_iter = 1024, 100
    n_workers = os.cpu_count()
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # Serial
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")
    
    # Parallel with varying chunk sizes
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
    for mult in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        n_chunks = mult * n_workers
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, tiny)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_workers=n_workers, n_chunks=n_chunks, pool=pool)
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        lif = n_workers * t_par / t_serial - 1
        print(f"{n_chunks:4d} chunks: {t_par:.3f}s, speedup={t_serial/t_par:.1f}x, lif={lif:.2f}")
"""

if __name__ == '__main__':
    N, max_iter = 8192, 100
    n_workers = os.cpu_count()
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # Serial
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")
    
    # Dask
    client = Client("tcp://10.92.1.223:8786")
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
    
    chunk_counts = [n_workers, n_workers*2, n_workers*4, n_workers*8, n_workers*16, n_workers*32, n_workers*64, n_workers*128]
    results = []

    print(f"{'n chunks':>8} | {'time (s)':>8} | {'vs 1x':>6} | {'speedup':>8} | {'LIF':>6}")
    print("-" * 50)

    for n_chunks in chunk_counts:
        run_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks)
            run_times.append(time.perf_counter() - t0)
        
        tp = statistics.median(run_times)
        speedup = t_serial / tp
        lif = (n_workers * tp / t_serial) - 1
        vs_1x = n_chunks / n_workers
        
        results.append((n_chunks, tp, lif))
        print(f"{n_chunks:8d} | {tp:8.3f} | {vs_1x:5.1f}x | {speedup:8.1f}x | {lif:6.2f}")

    """
    n_vals = [r[0] for r in results]
    t_vals = [r[1] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, t_vals, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.xlabel('Number of Chunks (n_chunks)')
    plt.ylabel('Wall Time (s)')
    plt.title('Dask Chunk Sweep: Wall Time vs n_chunks')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.axvline(best_n, color='r', linestyle='--', label=f'Optimal n={best_n}')
    plt.legend()
    
    plt.savefig('dask_sweep.png')
    """
    
    client.close()
            
            

