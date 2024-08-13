export GPUS_PER_NODE=8
export PIPELINE_SIZE=1
export DP_SIZE=1
NODE_SIZE=$(( $DP_SIZE *$PIPELINE_SIZE))
srun -p Intern5 --quotatype=spot --gres=gpu:$GPUS_PER_NODE -N $NODE_SIZE -n $NODE_SIZE bash train_moe.sh