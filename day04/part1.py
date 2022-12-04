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
    elf_a, elf_b = line.strip().split(",")
    a_start, a_end = elf_a.split("-")
    b_start, b_end = elf_b.split("-")
    inp.append([
        int(a_start),
        int(a_end),
        int(b_start),
        int(b_end)
    ])

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
    (len(inp), 1),   # global shape
    (1, 1),   # local shape
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
