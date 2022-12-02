__kernel void part1(
    __global const int* inp,
    __global int* out
) {
    int s = 0;
    int best = 0;
    for (int i = 0; i < get_global_size(0); i++) {
        s += inp[i];

        if (s > best) {
            best = s;
        }

        if (inp[i] == -1) {
            s = 0;
        }
    }
    out[0] = best;
}

__kernel void part2(
    __global const long* inp,
    __global long* out
) {
    long s = 0;
    long best = 0;
    long second_best = 0;
    long third_best = 0;
    for (long i = 0; i <= get_global_size(0); i++) {
        if (i == get_global_size(0) || inp[i] == -1) {
            if (s > best) {
                third_best = second_best;
                second_best = best;
                best = s;
            } else if (s > second_best) {
                third_best = second_best;
                second_best = s;
            } else if (s > third_best) {
                third_best = s;
            }
            s = 0;
        } else {
            s += inp[i];
        }
    }
    out[0] = best + second_best + third_best;
    out[1] = best;
    out[2] = second_best;
    out[3] = third_best;
}
