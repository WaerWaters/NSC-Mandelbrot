from multiprocessing import Pool
import random, time, os
from functools import reduce

def monte_carlo_chunk(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return inside_circle

def test_granularity(total_work, chunk_size, n_proc):
    n_chunks = total_work // chunk_size
    tasks = [chunk_size] * n_chunks
    t0 = time.perf_counter()
    if n_proc == 1:
        results = [monte_carlo_chunk(chunk) for chunk in tasks]
    else:
        with Pool(processes=n_proc) as pool:
            results = pool.map(monte_carlo_chunk, tasks)
    return time.perf_counter() - t0, 4 * sum(results) / total_work

"""
if __name__ == '__main__':
    total_work = 10_000_000
    n_proc = os.cpu_count() // 2
    chunk_sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    print(f"{'L':>12} | {'serial (s)':>12} | {'parallel (s)':>12}")
    for L in chunk_sizes:
        t_serial, _ = test_granularity(total_work, L, n_proc=1)
        t_parallel, pi = test_granularity(total_work, L, n_proc=n_proc)
        print(f"{L:12d} | {t_serial:12.4f} | {t_parallel:12.4f} (pi={pi:.4f})")
"""

#           L |   serial (s) | parallel (s)
#          10 |       2.1156 |       1.0261 (pi=3.1411)
#         100 |       1.7725 |       0.7896 (pi=3.1428)
#        1000 |       1.9707 |       0.7837 (pi=3.1417)
#       10000 |       1.9531 |       0.8559 (pi=3.1420)
#      100000 |       1.9814 |       0.8225 (pi=3.1415)
#     1000000 |       1.9883 |       0.8345 (pi=3.1414)

# Optimal chunk size would be around 1_000



#-------------------------------------------------------------------------------------------------------------------------------------------------------------

N = 1_000_000
data = [random.randint(10, 100) for _ in range(N)]

def subtract_seven(x):
    return x - 7

if __name__ == '__main__':
    # Part 1
    t0 = time.perf_counter()
    results_ser = reduce(lambda a, b: a + b, filter(lambda x: x % 2 == 1, map(subtract_seven, data)))
    t_serial = time.perf_counter() - t0


    # Part 2
    t0 = time.perf_counter()
    with Pool() as pool:
        mapped = pool.map(subtract_seven, data)
    results_par = reduce(lambda a, b: a + b, filter(lambda x: x % 2 == 1, mapped))
    t_parallel = time.perf_counter() - t0

    print(f"Serial: {t_serial:.4f}s results={results_ser}")
    print(f"Parallel: {t_parallel:.4f}s results={results_par}")
    print(f"Speedup: {t_serial / t_parallel:.2f}x")

#Serial: 0.1745s results=24294765
#Parallel: 1.3822s results=24294765
#Speedup: 0.13x

# map was faster than pool.map
