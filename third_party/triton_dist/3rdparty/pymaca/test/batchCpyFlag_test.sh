#!/bin/bash
torchrun --node_rank=0 --nproc_per_node=2 --nnodes=1 --rdzv_endpoint=127.0.0.1:23456 batchCpyFlag_test.py
