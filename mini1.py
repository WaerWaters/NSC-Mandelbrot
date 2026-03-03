import numpy
import cProfile, pstats
from mandelbrotFunc import compute_mandelbrot_naive, compute_mandelbrot_numpy
from mandelbrotBenchmark import benchmark

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
Tue Mar  3 12:07:37 2026    naive_profile.prof

         23046238 function calls (23043163 primitive calls) in 6.939 seconds

   Ordered by: cumulative time
   List reduced from 35 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    6.939    6.939 {built-in method builtins.exec}
        1    0.002    0.002    6.939    6.939 <string>:1(<module>)
        1    5.267    5.267    6.936    6.936 C:\Users\danie\NSC-Mandelbrot\mandelbrotFunc.py:4(compute_mandelbrot_naive)
 21959734    1.553    0.000    1.553    0.000 {built-in method builtins.abs}
  1049600    0.071    0.000    0.071    0.000 {method 'append' of 'list' objects}
     1025    0.001    0.000    0.046    0.000 <__array_function__ internals>:177(linspace)
4100/1025    0.007    0.000    0.044    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
     1025    0.014    0.000    0.044    0.000 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\core\function_base.py:23(linspace)
     1025    0.001    0.000    0.015    0.000 <__array_function__ internals>:177(any)
     1025    0.001    0.000    0.013    0.000 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\core\fromnumeric.py:2307(any)


Tue Mar  3 12:07:38 2026    numpy_profile.prof

         138 function calls (129 primitive calls) in 1.169 seconds

   Ordered by: cumulative time
   List reduced from 66 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.169    1.169 {built-in method builtins.exec}
        1    0.003    0.003    1.169    1.169 <string>:1(<module>)
        1    1.158    1.158    1.166    1.166 C:\Users\danie\NSC-Mandelbrot\mandelbrotFunc.py:30(compute_mandelbrot_numpy)
     14/5    0.004    0.000    0.007    0.001 {built-in method numpy.core._multiarray_umath.implement_array_function}
        2    0.000    0.000    0.004    0.002 <__array_function__ internals>:177(copyto)
        1    0.000    0.000    0.004    0.004 <__array_function__ internals>:177(zeros_like)
        1    0.000    0.000    0.004    0.004 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\core\numeric.py:76(zeros_like)
        1    0.000    0.000    0.003    0.003 <__array_function__ internals>:177(meshgrid)
        1    0.000    0.000    0.003    0.003 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\lib\function_base.py:4846(meshgrid)
        1    0.000    0.000    0.002    0.002 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\lib\function_base.py:4990(<listcomp>) 
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
Naive:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    6.939    6.939 {built-in method builtins.exec}
        1    0.002    0.002    6.939    6.939 <string>:1(<module>)
        1    5.267    5.267    6.936    6.936 C:\Users\danie\NSC-Mandelbrot\mandelbrotFunc.py:4(compute_mandelbrot_naive)
 21959734    1.553    0.000    1.553    0.000 {built-in method builtins.abs}
  1049600    0.071    0.000    0.071    0.000 {method 'append' of 'list' objects}
     1025    0.001    0.000    0.046    0.000 <__array_function__ internals>:177(linspace)
4100/1025    0.007    0.000    0.044    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
     1025    0.014    0.000    0.044    0.000 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\core\function_base.py:23(linspace)
     1025    0.001    0.000    0.015    0.000 <__array_function__ internals>:177(any)
     1025    0.001    0.000    0.013    0.000 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\core\fromnumeric.py:2307(any)

Numpy:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.169    1.169 {built-in method builtins.exec}
        1    0.003    0.003    1.169    1.169 <string>:1(<module>)
        1    1.158    1.158    1.166    1.166 C:\Users\danie\NSC-Mandelbrot\mandelbrotFunc.py:30(compute_mandelbrot_numpy)
     14/5    0.004    0.000    0.007    0.001 {built-in method numpy.core._multiarray_umath.implement_array_function}
        2    0.000    0.000    0.004    0.002 <__array_function__ internals>:177(copyto)
        1    0.000    0.000    0.004    0.004 <__array_function__ internals>:177(zeros_like)
        1    0.000    0.000    0.004    0.004 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\core\numeric.py:76(zeros_like)
        1    0.000    0.000    0.003    0.003 <__array_function__ internals>:177(meshgrid)
        1    0.000    0.000    0.003    0.003 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\lib\function_base.py:4846(meshgrid)
        1    0.000    0.000    0.002    0.002 C:\Users\danie\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\lib\function_base.py:4990(<listcomp>) 
"""

compute_mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100, display=False)

""" Profiling section
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