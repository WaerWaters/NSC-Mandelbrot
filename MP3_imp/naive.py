import numpy as np

def create_mandelbrot(res: int, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    """
    Generate a 2D grid representing the Mandelbrot set.

    Parameters
    ----------
    res : int
        The resolution of the grid (number of points along each axis).
    x_min : float
        The minimum value on the real axis.
    x_max : float
        The maximum value on the real axis.
    y_min : float
        The minimum value on the imaginary axis.
    y_max : float
        The maximum value on the imaginary axis.

    Returns
    -------
    NDArray[np.float64]
        A square 2D array of size (res, res) containing escape iteration counts.
    """
    x = np.linspace(start=x_min, stop=x_max, num=res)
    y = np.linspace(start=y_min, stop=y_max, num=res)
    grid = np.zeros((res, res))
    for i in range(0, res):
        for j in range(0, res):
            n = get_pixel(x=x[i], y=y[j], max_iter=100)
            grid[i][j] = n
    return grid

def get_pixel(x: float, y: float, max_iter: int) -> int:
    """
    Calculate the number of iterations before a point escapes the Mandelbrot set.

    Parameters
    ----------
    x : float
        The real component of the complex number.
    y : float
        The imaginary component of the complex number.
    max_iter : int
        The maximum number of iterations to perform.

    Returns
    -------
    int
        The iteration index at which |z| exceeded 2, or max_iter.
    """
    c = complex(x, y)
    z = 0
    for i in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return i
    return max_iter