#!/bin/bash
# ****************************** Exporting various arguments **********************************************
export TASK=$1                                   # "hellaswag"
export MODEL=$2                                  # "decapoda-research/llama-7b-hf"
export bs=$3                                     # 8
export sr=$4                                     # sparsity ratio for frozen parameters, default 0.5
export trial=$5                                  # trial number: 1, 2, 3, ... etc.
# *********************************************************************************************************

echo "task name:= " $TASK               
echo "model:= " $MODEL                  
echo "batch size:= " $bs                 
echo "sparsity ratio:= " $sr             
echo "trial number:= " $trial         

echo "NODE_RANK="$SLURM_NODEID
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

#export WANDB_DISABLED=true
#export NCCL_P2P_LEVEL=NVL

### bf16 renders better trianing performance
### baseline ft
torchrun --nproc_per_node=16 \
    --nnodes $SLURM_NNODES --node_rank=$SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    ../../run_clm_llama_lwcd_static_sparse.py \
    --model_name_or_path /nfs/projects/mbzuai/ext_hao.zhang/lanxiang/Sapling/outputs/decapoda-research/llama-7b-hf/piqa_bs_2/layerwise_condense_sparse_exhausive_sr0.625_trial_4_max_fro/checkpoint-24500 \
    --fp16 True \
    --total_layer_count 32 \
    --tie_breaker_strategy "activation" \
    --dataset_name ${TASK} \
    --per_device_eval_batch_size $bs \
    --sparsity_ratio $sr \
    --evaluation_strategy "epoch" \
    --max_budget 24 \
    --seed 42 \
    --do_eval \
    --save_steps 500 \
    --save_total_limit 1 \
    --output_dir ../../outputs/$MODEL/${TASK}_bs_${bs}/10_layers_checkpoint_sr${sr}_max_fro 2>&1 | tee ../logs/train/llama_7b_layerwise_condense_sparse_exhausive_${TASK}_bs_${bs}_sr${sr}_trial_${trial}_max_fro_eval_only.log