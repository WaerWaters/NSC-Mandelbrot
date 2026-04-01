import math, random, time, statistics, os
from multiprocessing import Pool

def estimate_pi_serial(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples

"""
if __name__ == '__main__':
    num_samples = 10_000_000
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        pi_estimate = estimate_pi_serial(num_samples)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"pi estimate: {pi_estimate:.6f} (error: {abs(pi_estimate-math.pi):.6f})")
    print(f"Serial time: {t_serial:.3f}s")

# pi estimate: 3.141402 (error: 0.000191)
# Serial time: 1.940s
"""
#------------------------------------------------------------------------------------------------------------------------------
def estimate_pi_chunk(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return inside_circle

def estimate_pi_parallel(num_samples, num_processes=4):
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / num_samples

"""
if __name__ == '__main__':
    num_samples = 10_000_000
    for num_proc in range(1, os.cpu_count() + 1):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            pi_est = estimate_pi_parallel(num_samples, num_proc)
            times.append(time.perf_counter() - t0)
        t = statistics.median(times)
        print(f"{num_proc:2d} workers: {t:.3f}s pi={pi_est:.6f}")

# All the workers does give around the same estimate
# I see a speedup from the first worker to the second worker
"""
#--------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    num_samples = 10_000_000
    cpu_limit = os.cpu_count()
    
    print(f"{'Workers':>8} | {'Time (s)':>10} | {'Speedup Sp':>10} | {'Efficiency Ep (%)':>18}")
    
    t_serial = 0

    for num_proc in range(1, cpu_limit + 1):
        times = []
        # Run 3 trials to get a stable median time
        for _ in range(3):
            t0 = time.perf_counter()
            _ = estimate_pi_parallel(num_samples, num_proc)
            times.append(time.perf_counter() - t0)
        
        t_p = statistics.median(times)
        
        # Set baseline from the first iteration (1 worker)
        if num_proc == 1:
            t_serial = t_p
        
        # Calculations
        speedup = t_serial / t_p
        efficiency = (speedup / num_proc) * 100
        
        # Print Tabulated Row
        print(f"{num_proc:8d} | {t_p:10.3f} | {speedup:10.2f} | {efficiency:17.1f}%")

# Workers |   Time (s) | Speedup Sp |  Efficiency Ep (%)
#       1 |      2.029 |       1.00 |             100.0%
#       2 |      1.140 |       1.78 |              89.0%
#       3 |      0.842 |       2.41 |              80.3%
#       4 |      0.767 |       2.65 |              66.2%
#       5 |      0.758 |       2.68 |              53.6%
#       6 |      0.788 |       2.58 |              42.9%
#       7 |      0.740 |       2.74 |              39.2%
#       8 |      0.747 |       2.72 |              33.9%

# Speedup is maximum at worker count 7
# it does plateau after worker count 7, it could be amdahl's law or hardware limits
# The serial fraction is approximately 25.9%