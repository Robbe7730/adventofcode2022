__kernel void part1(
    __global const long* inp,
    __global long* out
) {
    out[0] = 1337;
}

__kernel void part2(
    __global const long* inp,
    __global long* out
) {
    out[0] = 1338;
}
