__kernel void part1(
    __global const int* inp,
    __global int* out
) {
    int i = get_global_id(0);

    int their_choice = inp[i*2] - 'A';
    int our_choice = inp[(i*2)+1] - 'X';
    int diff = their_choice - our_choice;
    int score = our_choice + 1;

    if (diff == 0) {
        score += 3;
    } else if (diff == -1 || diff == 2) {
        score += 6;
    }

    atomic_add(&out[0], score);
}

__kernel void part2(
    __global const int* inp,
    __global int* out
) {
    int i = get_global_id(0);

    int their_choice = inp[i*2] - 'A';
    int expected_result = inp[(i*2)+1] - 'X';
    int score = expected_result * 3;
    int diff;

    if (expected_result == 1) {
        diff = their_choice;
    } else if (expected_result == 0) {
        diff = their_choice - 1;
    } else {
        diff = their_choice + 1;
    }

    if (diff < 0) {
        diff += 3;
    }

    if (diff >= 3) {
        diff -= 3;
    }

    score += diff + 1;

    atomic_add(&out[0], score);
}
