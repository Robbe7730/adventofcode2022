__kernel void part1(
    global const int* inp,
    global const int* lens,
    global int* out,
    global int* tmp,
    global int* tmp2
) {
    int letter = get_global_id(0);
    int backpack = get_global_id(1);
    int i = backpack * get_global_size(0) + letter;

    private int letter_i;

    if (inp[i] > 'Z') {
        letter_i = inp[i] - 'a';
    } else {
        letter_i = inp[i] - 'A' + 26;
    }

    if (inp[i] != 0) {
        if (letter < lens[backpack]) {
            if (letter < (lens[backpack] / 2)) {
                atomic_or(&tmp[backpack * 52 + letter_i], 1);
            } else {
                atomic_or(&tmp[backpack * 52 + letter_i], 2);
            }

            if (
                (tmp[backpack * 52 + letter_i] == 3) &&
                (atomic_cmpxchg(&tmp2[backpack * 52 + letter_i], 0, 1) == 0)
            ) {
                atomic_add(&out[0], letter_i + 1);
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (inp[i] != 0) {

    }
}

__kernel void part2(
    __global const long* inp,
    __global long* out
) {
    out[0] = 1338;
}
