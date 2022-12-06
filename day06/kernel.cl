__kernel void part1(
    __global const int* inp,
    __global int* out
) {
    atomic_cmpxchg(&out[0], 0, 10000);
    int i = get_global_id(0);

    if (i >= 3) {
        int found_mask = 0;
        int is_marker = 1;

        for (int offset = 0; offset < 4; offset++) {
            int mask = 1 << (inp[i-offset] - 'a');

            if ((found_mask & mask) != 0) {
                is_marker = 0;
            }

            found_mask = found_mask | mask;
        }

        if (is_marker == 1) {
            atomic_min(&out[0], i+1);
        }
    }
}

__kernel void part2(
    __global const int* inp,
    __global int* out
) {
    atomic_cmpxchg(&out[0], 0, 10000);
    int i = get_global_id(0);

    if (i >= 14) {
        int found_mask = 0;
        int is_marker = 1;

        for (int offset = 0; offset < 14; offset++) {
            int mask = 1 << (inp[i-offset] - 'a');

            if ((found_mask & mask) != 0) {
                is_marker = 0;
            }

            found_mask = found_mask | mask;
        }

        if (is_marker == 1) {
            atomic_min(&out[0], i+1);
        }
    }
}
