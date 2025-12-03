#!/bin/bash

set -e # stop on any error from now on

# Define the list of number pairs as strings
number_pairs="2,4 2,8 4,8 2,16 4,16 8,16"

# Split the string into an array based on spaces
IFS=' ' read -r -a pairs_array <<< "$number_pairs"

bs=(
     128
     #512
     #1024
     #2048
)

# Loop through the array of pairs
for pair in "${pairs_array[@]}"; do
    # Split the current pair into two numbers based on the comma
    IFS=',' read -r -a numbers <<< "$pair"

    # Access the numbers
    topk=${numbers[0]}
    e=${numbers[1]}
        for b in "${bs[@]}"; do
            echo
	    echo "batch_size: $b, topk: $topk, N_expert: $e"
            #ls -1 benchmarks/workspace
	    rm benchmarks/workspace/*
            python3 benchmarks/generator.py -b 10 -m $b --topk $topk -e $e
            sleep 1
            python3 benchmarks/bench.py
            sleep 1
            echo
	done
done
