import re
import os
import argparse
import subprocess
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--id', type=int, default=0)
    parser.add_argument('-f',
                        type=str,
                        default="benchmarks/workspace",
                        help='folder with problems')

    args = parser.parse_args()
    return args


def process_path(path: str):
    problems = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")

    # Check if it's a directory
    if os.path.isdir(path):
        print(f"Solve problems in directory: {path}")
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                problems.append(file_path)

    # Check if it's a file
    elif os.path.isfile(path):
        # print(f"Contents of file: {path}")
        # with open(path, 'r') as file:
        #     print(file.read())
        problems.append(path)

    else:
        raise RuntimeError(f"{path} is not a regular file or directory.")

    return problems


def get_gflops(output: str):
    gflops = []

    # Use regular expression to find the GFLOPs values
    gflops_pattern = r'Grouped\s+GFLOPs:\s+(\d+)'
    gflops_matches = re.findall(gflops_pattern, output)

    for match in gflops_matches:
        gflops.append(int(match))

    ### Print the output (stdout)
    # print("Standard Output:")
    # print(output)
    # print(gflops)

    return gflops


def main():
    args = parse_args()

    # problems = ['./build/problems.txt']
    problems = process_path(args.f)
    print(f'problems: {problems}')
    print('=======================')

    perf = {
        'device': [],
        'precompute': [],
        'streamk': [],
    }

    for problem in problems:
        # command = ['./build/examples/24_gemm_grouped/streamk_grouped_gemm', '--verbose=true', '--benchmark=./build/problems.txt']

        command = [
            './build/examples/24_gemm_grouped/streamk_grouped_gemm',
            '--verbose=true', f'--benchmark={problem}'
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            gflops = get_gflops(result.stdout)

            assert (
                len(gflops) == 3
            ), f'Error: {gflops} \n\n {problem} \n\n {result.stdout} \n\n {result.stderr}'
            perf['device'].append(gflops[0])
            perf['precompute'].append(gflops[1])
            perf['streamk'].append(gflops[2])

            # Print any errors (stderr)
            if result.stderr:
                print("Standard Error:")
                print(result.stderr)
                raise subprocess.CalledProcessError(
                    returncode=result.returncode,
                    cmd=command,
                    stderr=result.stderr)

        except subprocess.CalledProcessError as e:
            # Handle the case where the command returns a non-zero exit code
            print(
                f"An error occurred while running the command: {e} \n\n {e.cmd} \n\n {e.stderr} \n\n {e.stdout}"
            )
            raise e

    # normalized gflops
    norm = {
        'device': [i / j for i, j in zip(perf['device'], perf['precompute'])],
        'precompute': [],
        'streamk':
        [i / j for i, j in zip(perf['streamk'], perf['precompute'])],
    }

    print('[Normalized GFLOPs]: ')
    summ = sum(norm['device']) / len(norm['device'])
    geomean = np.power(np.prod(norm['device']), 1 / len(norm['device']))
    print(f'device - avg: {summ}, geomean: {geomean}')

    summ = sum(norm['streamk']) / len(norm['streamk'])
    geomean = np.power(np.prod(norm['streamk']), 1 / len(norm['streamk']))
    print(f'streamk - avg: {summ}, geomean: {geomean}')


if __name__ == '__main__':
    main()
