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


#cProfile.run('compute_mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100, display=False)', 'naive_profile.prof')
#cProfile.run('compute_mandelbrot_numpy(-2, 1, -1.5, 1.5, 1024, 1024, 100, display=False)', 'numpy_profile.prof')

#for name in ('naive_profile.prof', 'numpy_profile.prof'):
#    stats = pstats.Stats(name)
#    stats.sort_stats('cumulative')
#    stats.print_stats(10)

    
"""
Lecture 3, Slide 27

Q: Which function takes most total time?
A: Naive takes the most time to run

Q: Are there functions called surprisingly many times?
A: Some are called more than 20 million times

Q: How does the NumPy profile compare to naive?
A: Still faster, fewer calls, has more functions than Naive

cProfile output:
Go to profileOutputs.py, Lecture 3, Slide 27, to see output
"""

"""
Lecture 3, Slide 28

Q: Which function takes most total time?
A: compute_mandelbrot_naive, is the function that takes the longest

Q: Are there functions called surprisingly many times?
A: yeah, some are still called more than 20 million times

Q: How does NumPy profile compare to naive?
A: still faster, fewer calls, has more functions than Naive

Q: Where does NumPy spend its time?
A: it spends 1.158 seconds on the compute_mandelbrot_numpy function, and next most on {built-in method numpy.core._multiarray_umath.implement_array_function} with 0.004 seconds

Summary Table:
Go to profileOutput.py, Lecture 3, Slide 28, to see summary table
"""

#compute_mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100, display=False)

""" Profiling section
Lecture 3, Slide 31

Q: cProfile on naive vs NumPy: How many functions appear in each profile?
A: Naive has 35 functions, and Numpy has 66 functions
Q: What does this difference tell you about where the work actually happens?
A: Naive spends its time in the interpreter, where Numpy spends its time preparing the data, so the heavy lifting can be done by pre-compiled C code

Q: Line profiler on naive: Which lines dominate runtime?
A: The lines in the innermost loop, where we calculate z with 36.7s, and check if abs(z) is greater than 2 with 38.3s
Q: What fraction of total time is spent in the inner loop?
A: The fraction is 925/1000

Q: Based on your profiling results: why is NumPy faster than naive Python?
A: Because Numpy doesn't have multiple nested loops.

Q: What would you need to change to make the naive version faster?
A: Avoid using abs(), and pre-allocate the list instead of using append, because it can make python re-allocate memory for the entire list
"""

""" Milestone 3 """
#_ = compute_mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 1024, 1024, 100)
#_ = compute_mandelbrot_hybrid(-2, 1, -1.5, 1.5, 1024, 1024, 100)
#t_full = bench(compute_mandelbrot_naive_numba, -2, 1, -1.5, 1.5, 1024, 1024, 100)
#t_hybrid = bench(compute_mandelbrot_hybrid, -2, 1, -1.5, 1.5, 1024, 1024, 100)

#print(f"Hybrid: {t_hybrid:.3f}s")
#print(f"Fully compiled: {t_full:.3f}s")
#print(f"Ratio: {t_hybrid/t_full:.1f}x")


""" Milestone 4 """
for dtype in [numpy.float16, numpy.float32, numpy.float64]:
    t0 = time.perf_counter()
    compute_mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=dtype)
    print(f"{dtype.__name__}: {time.perf_counter()-t0:.3f}s")

r16 = compute_mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=numpy.float16)
r32 = compute_mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=numpy.float32)
r64 = compute_mandelbrot_numba_typed(-2 ,1 ,-1.5 ,1.5, 1024, 1024, 100, dtype=numpy.float64)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, result, title in zip(axes, [r16, r32, r64], ['float16', 'float32', 'float64 (ref)']):
    ax.imshow(result, cmap='hot')
    ax.set_title(title); ax.axis('off')
plt.savefig('precision_comparison.png', dpi=150)
print(f"Max diff float32 vs float64: {numpy.abs(r32 - r64).max()}")
print(f"Max diff float16 vs float64: {numpy.abs(r16 - r64).max()}")

"""
Lecture 3, Slide 44

Q: Speed: Does float32 actually run faster than float64 on your hardware?
A: somes times float32 takes between 0.002-0.023 seconds less than float64, but apart from that, no
Q: By how much?
A: 0.002-0.023 seconds

Q: float16: Try it with NumPy — is it faster than float32?
A: Yes it's faster by 0.011 seconds. float16 = 1.138, float32 = 1.149

Q: Visual quality: Zoom in on a detailed region of the Mandelbrot set. Can you see artefacts with float16?
A: Yes, I can see artifacts
Q: What about float32?
A: Yes, I can also see artifacts using float32

Q: Recommendation: Based on what you observe, which precision would you choose for production use, and why?
A: I would use float32 because its more than 4 times faster than float16 and is around the same speed as float64, and the accuracy is the same across float16, float32 and float64
"""

""" Data Type Section
Measured runtimes:
float16: 1.166s
float32: 0.240s
float64: 0.240s

Side-by-Side images:
Check precision_comparison.png for side-by-side image

Recommendation:
I would use float32 because its more than 4 times faster than float16 and is around the same speed as float64, and the accuracy is the same across float16, float32 and float64

"""
