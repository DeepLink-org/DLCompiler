import os
import random
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--id', type=int, default=0)
    parser.add_argument('-m', type=int, default=2048)
    parser.add_argument('-n', type=int, default=14336)
    parser.add_argument('-k', type=int, default=4096)
    parser.add_argument('-e', type=int, default=8, help='number of experts')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--topk', type=int, default=2)

    parser.add_argument('-b',
                        type=int,
                        default=1,
                        help='batch generate problem')

    args = parser.parse_args()
    return args


def generate_random_partition(n, k=16):
    if n < k:
        raise ValueError("n must be greater than or equal to k")

    # Initialize the list with 1s
    partition = [1] * k

    # Distribute the remaining n-k among the k bins
    remaining = n - k
    for _ in range(remaining):
        # Randomly select a bin and increment it
        bin_index = random.randint(0, k - 1)
        partition[bin_index] += 1

    return partition


def main():
    args = parse_args()
    random.seed(args.seed)

    n, k = args.n, args.k
    total = args.topk * args.m

    for pid in range(args.b):
        ms = generate_random_partition(total, args.e)

        text = ""
        for i, m in enumerate(ms):
            line = f"{i} {m}x{n}x{k}\n"
            text += line

        with open(f'benchmarks/workspace/problem{pid}.txt', 'w') as file:
            file.write(text)


if __name__ == '__main__':
    main()
