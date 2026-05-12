#!/bin/bash
#run single device op test
bash ./launch_single_device.sh ./test_distributed_wait.py --case correctness
python test_common_ops.py
python test_language_extra.py
python test_memory_ops.py
#run multi-devices test
bash ./run_mxshmem_api_test.sh
bash ./launch.sh ./test_ag_gemm_intra_node.py --case correctness_no_tma
#run multi-machines test
bash ./launch_inter_node.sh ./test_ag_gemm_inter_node.py
