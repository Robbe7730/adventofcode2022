#!/usr/bin/env python3

import numpy as np
import pyopencl as cl

import sys

np.set_printoptions(linewidth=200)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# 1. Read input
inp = []
lens = []
for line in sys.stdin:
    line = line.strip()
    inp.append([ord(x) for x in line])
    lens.append(len(line))

inp = [x + [0] * (52 - len(x)) for x in inp]

# 2. Prepare numpy arrays
inp_np = np.array(inp, dtype="int32")
lens_np = np.array(lens, dtype="int32")
out_np = np.zeros((1,), dtype="int32")
tmp1_np = np.zeros((len(inp), 52), dtype="int32")
tmp2_np = np.zeros((len(inp), 52), dtype="int32")

# 3. Prepare buffers
mf = cl.mem_flags
inp_buffer = cl.Buffer(
    ctx,
    mf.READ_ONLY | mf.COPY_HOST_PTR,
    hostbuf=inp_np
)
lens_buffer = cl.Buffer(
    ctx,
    mf.READ_ONLY | mf.COPY_HOST_PTR,
    hostbuf=lens_np
)
out_buffer = cl.Buffer(
    ctx,
    mf.COPY_HOST_PTR,
    hostbuf=out_np
)
tmp1_buffer = cl.Buffer(
    ctx, mf.COPY_HOST_PTR,
    hostbuf=tmp1_np
)
tmp2_buffer = cl.Buffer(
    ctx, mf.COPY_HOST_PTR,
    hostbuf=tmp2_np
)

# 4. Create kernel
with open("kernel.cl", "r") as kernel_file:
    prg = cl.Program(ctx, kernel_file.read()).build()

# 5. Run kernel
knl = prg.part1

knl(
    queue,           # queue
    (52, len(inp)),   # global shape
    (1, 1),          # local shape
    inp_buffer,      # args...
    lens_buffer,
    out_buffer,
    tmp1_buffer,
    tmp2_buffer,
)

# 6. Get result
cl.enqueue_copy(
    queue,
    out_np,
    out_buffer
)
print(out_np)
