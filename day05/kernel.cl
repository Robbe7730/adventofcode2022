__kernel void part1(
    __global int* stack,
    __global const int* moves,
    __global int* out,
    __global const int* meta
) {
    int width = meta[0];
    int height = meta[1];
    int num_moves = meta[2];

    for (int move = 0; move < num_moves; move++) {
        int amount = moves[move*3];
        int from = moves[(move*3)+1] - 1;
        int to = moves[(move*3)+2] - 1;

        int from_top = 0;
        while (stack[(from_top*width)+from] == 0 && from_top < height) {
            from_top++;
        }

        int to_top = 0;
        while (stack[(to_top*width)+to] == 0 && to_top < height) {
            to_top++;
        }

        for (int i = 0; i < amount; i++) {
            stack[(to_top-1)*width + to] = stack[from_top*width + from];
            stack[from_top*width + from] = 0;

            from_top++;
            to_top--;
            
            if (from_top > height) {
                break;
            }
        }
    }

    for (int col = 0; col < width; col++) {
        int top = 0;
        while (stack[(top*width)+col] == 0 && top < height) {
            top++;
        }

        out[col] = stack[top*width + col];
    }
}

__kernel void part2(
    __global int* stack,
    __global const int* moves,
    __global int* out,
    __global const int* meta
) {
    int width = meta[0];
    int height = meta[1];
    int num_moves = meta[2];

    for (int move = 0; move < num_moves; move++) {
        int amount = moves[move*3];
        int from = moves[(move*3)+1] - 1;
        int to = moves[(move*3)+2] - 1;

        int from_top = 0;
        while (stack[(from_top*width)+from] == 0 && from_top < height) {
            from_top++;
        }

        int to_top = 0;
        while (stack[(to_top*width)+to] == 0 && to_top < height) {
            to_top++;
        }

        for (int i = 0; i < amount; i++) {
            stack[(to_top-(amount-i))*width + to] = stack[from_top*width + from];
            stack[from_top*width + from] = 0;

            from_top++;
            
            if (from_top > height) {
                break;
            }
        }
    }

    for (int col = 0; col < width; col++) {
        int top = 0;
        while (stack[(top*width)+col] == 0 && top < height) {
            top++;
        }

        out[col] = stack[top*width + col];
    }
}
