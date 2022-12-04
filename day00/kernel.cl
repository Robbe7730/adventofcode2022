__kernel void part1(
    __global const int* inp,
    __global int* out
) {
    out[0] = 1337;
}

__kernel void part2(
    __global const int* inp,
    __global int* out
) {
    out[0] = 1338;
}
