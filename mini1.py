import numpy
import cProfile, pstats
from mandelbrotFunc import compute_mandelbrot_naive, compute_mandelbrot_numpy, compute_mandelbrot_naive_numba, compute_mandelbrot_hybrid
from mandelbrotBenchmark import benchmark, bench

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


_ = compute_mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 1024, 1024, 100)
_ = compute_mandelbrot_hybrid(-2, 1, -1.5, 1.5, 1024, 1024, 100)
t_full = bench(compute_mandelbrot_naive_numba, -2, 1, -1.5, 1.5, 1024, 1024, 100)
t_hybrid = bench(compute_mandelbrot_hybrid, -2, 1, -1.5, 1.5, 1024, 1024, 100)

print(f"Hybrid: {t_hybrid:.3f}s")
print(f"Fully compiled: {t_full:.3f}s")
print(f"Ratio: {t_hybrid/t_full:.1f}x")


