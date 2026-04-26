import numpy as np
import pyopencl as cl
import time, matplotlib.pyplot as plt



KERNEL_SRC = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    
    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double z_real = 0.0, z_imag = 0.0;
    int count = 0;
    while (count < max_iter && z_real*z_real + z_imag*z_imag <= 4.0) {
        double tmp = z_real*z_real - z_imag*z_imag + c_real;
        z_imag = 2.0 * z_real * z_imag + c_imag;
        z_real = tmp;
        count++;
    }
    result[row * N + col] = count;
}"""

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
prog = cl.Program(ctx, KERNEL_SRC).build()
mandel_kernel = cl.Kernel(prog, 'mandelbrot')

N, MAX_ITER = 2048, 200
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

image = np.zeros((N, N), dtype=np.int32)
image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

mandel_kernel(
    queue, (N, N), None, image_dev,
    np.float64(X_MIN), np.float64(X_MAX),
    np.float64(Y_MIN), np.float64(Y_MAX),
    np.int32(N), np.int32(MAX_ITER)
)
cl.enqueue_copy(queue, image, image_dev)
queue.finish()


# warm up
mandel_kernel(
    queue, (64, 64), None, image_dev,
    np.float64(X_MIN), np.float64(X_MAX),
    np.float64(Y_MIN), np.float64(Y_MAX),
    np.int32(64), np.int32(MAX_ITER))
queue.finish()

# Time real run
t0 = time.perf_counter()
mandel_kernel(
    queue, (N, N), None, image_dev,
    np.float64(X_MIN), np.float64(X_MAX),
    np.float64(Y_MIN), np.float64(Y_MAX),
    np.int32(N), np.int32(MAX_ITER))
queue.finish()
elapsed = time.perf_counter() - t0

cl.enqueue_copy(queue, image, image_dev)
queue.finish()

print(f"GPU {N}x{N}: {elapsed*1e3:.1f} ms")
plt.imshow(image, cmap='hot', origin='lower'); plt.axis('off')
plt.savefig("mandelbrot_gpu.png", dpi=150, bbox_inches='tight')