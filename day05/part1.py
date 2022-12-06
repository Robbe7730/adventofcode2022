#!/usr/bin/env python3

import numpy as np
import pyopencl as cl

import sys

np.set_printoptions(linewidth=200)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# 1. Read input
stack = []
moves = []
read_buffer = False
max_height = 0
for line in sys.stdin:
    if line == "\n":
        read_buffer = True
        continue

    if read_buffer:
        _, amount, _, src, _, dest = line.strip().split(" ")
        moves.append([int(amount), int(src), int(dest)])
    elif "[" in line:
        row = []
        for i in range(1, len(line), 4):
            if line[i] == " ":
                row.append(0)
            else:
                row.append(ord(line[i]))
                max_height += 1
        stack.append(row)

for _ in range(len(stack), max_height):
    stack = [[0] * len(stack[0])] + stack

# 2. Prepare numpy arrays
stack_np = np.array(stack, dtype="int32")
moves_np = np.array(moves, dtype="int32")
out_np = np.zeros((len(stack[0]),), dtype="int32")
meta_np = np.array([len(stack[0]), len(stack), len(moves)], dtype="int32")

# 3. Prepare buffers
mf = cl.mem_flags
stack_buffer = cl.Buffer(
    ctx,
    mf.COPY_HOST_PTR,
    hostbuf=stack_np
)

moves_buffer = cl.Buffer(
    ctx,
    mf.READ_ONLY | mf.COPY_HOST_PTR,
    hostbuf=moves_np
)

out_buffer = cl.Buffer(
    ctx,
    mf.COPY_HOST_PTR,
    hostbuf=out_np
)

meta_buffer = cl.Buffer(
    ctx,
    mf.READ_ONLY | mf.COPY_HOST_PTR,
    hostbuf=meta_np
)

# 4. Create kernel
with open("kernel.cl", "r") as kernel_file:
    prg = cl.Program(ctx, kernel_file.read()).build()

# 5. Run kernel
knl = prg.part1

knl(
    queue,                  # queue
    (1, 1),   # global shape
    (1, 1),   # local shape
    stack_buffer,             # args...
    moves_buffer,
    out_buffer,
    meta_buffer
)

# 6. Get result
cl.enqueue_copy(
    queue,
    out_np,
    out_buffer
)
print("".join([chr(x) for x in out_np]))
