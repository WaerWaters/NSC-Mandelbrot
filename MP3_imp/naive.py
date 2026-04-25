import numpy as np

def create_mandelbrot(res, x_min, x_max, y_min, y_max):
    x = np.linspace(start=x_min, stop=x_max, num=res)
    y = np.linspace(start=y_min, stop=y_max, num=res)
    grid = np.zeros((res, res))
    for i in range(0, res):
        for j in range(0, res):
            n = get_pixel(x=x[i], y=y[j], max_iter=100)
            grid[i][j] = n
    return grid

def get_pixel(x, y, max_iter):
    c = complex(x, y)
    z = 0
    for i in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return i
    return max_iter