import matplotlib.pyplot as plt

# Data
labels = [
    'Naive Python', 'NumPy', 'Dask Cluster*', 'Dask Local', 
    'Numba f32', 'Numba f64', 'GPU f64', 'Multiprocessing', 'GPU f32'
]
times = [
    3.9399, 1.162, 2.053, 0.08, 
    0.053, 0.051, 0.0245, 0.023, 0.0026
]

# Define colors - GPU is Green, Numba is Blue, Naive is Red
colors = [
    '#e74c3c', '#34495e', '#9b59b6', '#f39c12', 
    '#3498db', '#3498db', '#2ecc71', '#f1c40f', '#2ecc71'
]

plt.figure(figsize=(12, 7))

# Create bars
bars = plt.bar(labels, times, color=colors, edgecolor='black', alpha=0.8)

# Apply a distinct texture (hatch) to the Dask Cluster bar to show it's a different scale
bars[2].set_hatch('//')
bars[2].set_edgecolor('white')

plt.yscale('log')
plt.grid(axis='y', which='both', linestyle='--', alpha=0.4)

# Labels
plt.title('MP3 Performance: Execution Time', fontsize=15, pad=20)
plt.ylabel('Seconds (Log Scale)', fontsize=12)
plt.xticks(rotation=30, ha='right')

# Annotate raw values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add context box
plt.annotate('* Dask Cluster run on 8192x8192\n   All others run on 1024x1024\n* Not all implementations are run\n   on the same hardware', 
             xy=(0.75, 0.85), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="white", ec="gray"))

plt.tight_layout()
plt.savefig('mp3_performance_comparison.png', dpi=300)
plt.show()