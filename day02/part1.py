#!/usr/bin/env python3

import numpy as np
import pyopencl as cl

import sys

np.set_printoptions(linewidth=200)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# 1. Read input
inp = []
for line in sys.stdin:
    inp.append([ord(x) for x in line.strip().split()])
# 2. Prepare numpy arrays
inp_np = np.array(inp, dtype="int32")
out_np = np.zeros((1,), dtype="int32")

# 3. Prepare buffers
mf = cl.mem_flags
inp_buffer = cl.Buffer(
    ctx,
    mf.READ_ONLY | mf.COPY_HOST_PTR,
    hostbuf=inp_np
)

out_buffer = cl.Buffer(
    ctx,
    mf.COPY_HOST_PTR,
    hostbuf=out_np
)

# 4. Create kernel
with open("kernel.cl", "r") as kernel_file:
    prg = cl.Program(ctx, kernel_file.read()).build()

# 5. Run kernel
knl = prg.part1

knl(
    queue,                  # queue
    (inp_np.shape[0],),   # global shape
    (1,),   # local shape
    inp_buffer,             # args...
    out_buffer
)

# 6. Get result
cl.enqueue_copy(
    queue,
    out_np,
    out_buffer
)
print(out_np)

# 14893 --> Too high
# 10903 --> Too low
