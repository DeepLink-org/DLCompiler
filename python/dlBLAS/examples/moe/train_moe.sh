WORLD_SIZE=$(( $SLURM_JOB_NUM_NODES ))
export MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
export MASTER_PORT=20031
RANK=$(( $SLURM_PROCID ))

if [ -z "$GPUS_PER_NODE" ]; then
  GPUS_PER_NODE=8
fi

run_cmd="torchrun --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --nproc_per_node=$GPUS_PER_NODE train.py --config ./configs/1.8B_MoE64.py \
  --launcher torch \
  --profiling"

echo $run_cmd
eval $run_cmd

set +x