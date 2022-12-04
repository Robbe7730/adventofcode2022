__kernel void part1(
    __global const int* inp,
    __global int* out
) {
    int i = get_global_id(0);
    int a_start = inp[i*4];
    int a_end = inp[(i*4)+1];
    int b_start = inp[(i*4)+2];
    int b_end = inp[(i*4)+3];

    if ((a_start >= b_start && a_end <= b_end) ||
        (a_start <= b_start && a_end >= b_end)) {
        atomic_add(&out[0], 1);
    }
}

__kernel void part2(
    __global const int* inp,
    __global int* out
) {
    int i = get_global_id(0);
    int a_start = inp[i*4];
    int a_end = inp[(i*4)+1];
    int b_start = inp[(i*4)+2];
    int b_end = inp[(i*4)+3];

    if (!((a_end < b_start) || (b_end < a_start))) {
        atomic_add(&out[0], 1);
    }
}
