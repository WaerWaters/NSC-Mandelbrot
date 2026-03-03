""" Lecture 3, Slide 27
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





""" Lecture 3, Slide 28
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