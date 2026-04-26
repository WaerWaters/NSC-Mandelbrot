#!/usr/bin/env python3
"""
opencl_template.py — Starting point for writing a PyOpenCL kernel.

Replace the vector-add kernel below with your own kernel.
The six steps are the same for every OpenCL program.
"""

import time
import numpy as np
import pyopencl as cl

VEC_SIZE = 50_000

# --- Step 1: create context and command queue ---
ctx   = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
print(f"Device: {ctx.devices[0].name}")

# --- Step 2: prepare host arrays ---
a_host      = np.random.rand(VEC_SIZE).astype(np.float32)
b_host      = np.random.rand(VEC_SIZE).astype(np.float32)
result_host = np.empty_like(a_host)

# --- Step 3: allocate device buffers and copy input data ---
mf       = cl.mem_flags
a_dev    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=a_host)
b_dev    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=b_host)
res_dev  = cl.Buffer(ctx, mf.WRITE_ONLY, a_host.nbytes)

# --- Step 4: compile the kernel ---
# To load from a separate file instead: KERNEL_SRC = open("kernel.cl").read()
KERNEL_SRC = """
__kernel void sum(
    __global const float *a,
    __global const float *b,
    __global       float *result)
{
    int gid = get_global_id(0);
    result[gid] = a[gid] + b[gid];
}
"""
prog = cl.Program(ctx, KERNEL_SRC).build()

# --- Step 5: launch the kernel ---
t0 = time.perf_counter()
prog.sum(queue, a_host.shape, None, a_dev, b_dev, res_dev)
queue.finish()
elapsed = time.perf_counter() - t0

# --- Step 6: copy result back to host ---
cl.enqueue_copy(queue, result_host, res_dev)
queue.finish()

# Verify and report
print(f"Elapsed:  {elapsed*1000:.3f} ms")
print(f"Correct:  {np.allclose(result_host, a_host + b_host)}")

# Q: What value does get global id(0) return for work-item 42 if global size = (50000,)?
# A: It returns 42, since the global id corresponds to the index of the work-item in the global execution space.

# Q: What happens if you launch with (49999,) instead of (N,)?
# A: If you launch with (49999,) instead of (N,), the kernel will only execute for the first 49999 work-items, and the last element of the result array will not be computed, leading to incorrect results for that element.

# Q: Why must the result buffer use WRITE ONLY and inputs use READ ONLY?
# A: The result buffer must use WRITE_ONLY to indicate that it will only be written to by the kernel, which can help optimize memory access. The input buffers use READ_ONLY to indicate that they will only be read from, which also allows for potential optimizations and ensures that the kernel does not attempt to modify the input data.