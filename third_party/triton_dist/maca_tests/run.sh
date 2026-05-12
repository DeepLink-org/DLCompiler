#!/bin/bash
set -x
# set mxshmem home to find mxshmem library
if [[ -n ${MXSHMEM_HOME} ]]; then
    export LD_LIBRARY_PATH=${MXSHMEM_HOME}/build/src:$LD_LIBRARY_PATH
    echo "LIBMXSHMEM_HOST PATH is set to: ${MXSHMEM_HOME}/build/src"
fi

export MXSHMEM_BOOTSTRAP=UID
export MXSHMEM_IB_ENABLE_IBRC=0

test_lists=(
    "test_mxshmem_api_putmem_signal.py"
    "test_mxshmem_api_remote_ptr.py"
    "test_mxshmem_api_signal_op.py"
    "test_mxshmem_api_basic.py"
)

nproc_per_node=2
nnodes=${WORKER_NUM:=1}
node_rank=${WORKER_ID:=0}

master_addr="127.0.0.1"
master_port="23459"

additional_args="--rdzv_endpoint=${master_addr}:${master_port}"

CMD_PREFIX="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${additional_args}"

for test in ${test_lists[@]}; do
    CMD="${CMD_PREFIX} ${test}"
    echo ${CMD}
    ${CMD}
    if [[ $? != 0 ]]; then
        exit 1
    fi
done

exit 0
