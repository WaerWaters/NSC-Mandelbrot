import dask, random, time, statistics
from dask import delayed
from dask.distributed import Client, LocalCluster

def monte_carlo_chunk(n_samples):
    inside = 0
    for _ in range(n_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return inside

total, n_chunks = 1_000_000, 128
samples = total // n_chunks

# Serial version
t0 = time.perf_counter()
results = [monte_carlo_chunk(samples) for _ in range(n_chunks)]
t_serial = time.perf_counter() - t0
print(f"Serial: {t_serial:.3f}s, pi={4 * sum(results) / total:.4f}")

# Dask delayed version
tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]
t0 = time.perf_counter()
results = dask.compute(*tasks)
t_dask = time.perf_counter() - t0
print(f"Dask: {t_dask:.3f}s, pi={4 * sum(results) / total:.4f}")

# Smaller chunks make Dask faster, and bigger chunks make it slower

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)

    print(f"Dashboard: {client.dashboard_link}")

    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]
    results = dask.compute(*tasks)

    cluster.scale(4); client.wait_for_workers(4)
    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]
    results = dask.compute(*tasks)

    client.close(); cluster.close()

# Link to dashboard doesn't work







