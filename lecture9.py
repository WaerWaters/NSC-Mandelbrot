import numpy as np
import matplotlib.pyplot as plt

# E1
# Naive version
def get_pixel(c, max_iter):
    z = 0j
    for i in range(max_iter):
        z = z**2 + c
        if abs(z) > 2:
            return i
    return max_iter

def test_pixel():
    c = complex(-1, 0)
    max_iter = 100
    result = get_pixel(c, max_iter)
    assert result == max_iter, f"Point c=-1 should not escape as it oscillate between 0 and 1, but got{result}"

# test_pixel()


# E2
def compute_mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter, display=False):
    """
    Generate a 2D grid of Mandelbrot set.

    Parameters
    ----------
    xmin, xmax : float
        The minimum and maximum boundaries of the real axis (horizontal).
    ymin, ymax : float
        The minimum and maximum boundaries of the imaginary axis (vertical).
    width : int
        The number of points to sample along the real axis.
    height : int
        The number of points to sample along the imaginary axis.
    max_iter : int
        The maximum number of iterations to check for divergence before 
        assuming a point is bounded within the set.
    display : bool, optional
        If True, renders and saves a heatmap of the results using matplotlib.
        Defaults to False.

    Returns
    -------
    list of list of int
        A 2D nested list (width x height) where each integer represents 
        the iteration count at which the point escaped the radius of 2. 
        Points that never escaped contain the value `max_iter`.

    Examples
    --------
    >>> data = compute_mandelbrot_naive(-2.5, 1.0, -1.25, 1.25, 1024, 1024, 100)
    """
    saved_iter = []
    for x in numpy.linspace(xmin, xmax, width):
        temp = []
        for y in numpy.linspace(ymin, ymax, height):
            c = complex(x, y)
            z = 0
            for i in range(max_iter):
                z = z**2 + c
                if abs(z) > 2:
                    temp.append(i)
                    break
            else:
                temp.append(max_iter)
        saved_iter.append(temp)
    
    if display:
        plt.imshow(saved_iter, cmap="hot")
        plt.title("Mandelbrot Set")
        plt.colorbar()
        plt.savefig("mandelbrot.png")
        plt.show()
   
    return saved_iter










