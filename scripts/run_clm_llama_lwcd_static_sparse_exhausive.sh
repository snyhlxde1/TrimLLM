TASK=$1               # "hellaswag"
MODEL=$2              # "decapoda-research/llama-7b-hf"
bs=$3                 # 8
sr=$4                 # sparsity ratio for frozen parameters, default 0.5
trial=$5              # trial number: 1, 2, 3, ... etc.

export WANDB_DISABLED=true
#export NCCL_P2P_LEVEL=NVL

mkdir -p logs/train/

### baseline ft
torchrun --nproc_per_node=8 \
    --master_port 20688 \
    run_clm_llama_lwcd_static_sparse.py \
    --bf16 True \
    --model_name_or_path $MODEL \
    --total_layer_count 32 \
    --tie_breaker_strategy "activation" \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate 2e-5 \
    --num_train_epochs 41 \
    --sparsity_ratio $sr \
    --condense_epoch 1 \
    --evaluation_strategy "epoch" \
    --max_budget 16 \
    --seed 42 \
    --do_train \
    --do_eval \
    --save_steps 50000 \
    --save_total_limit 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --output_dir outputs/$MODEL/${TASK}_bs_${bs}/layerwise_condense_sparse_exhausive_sr${sr}_trial_${trial}_min_fro 2>&1 | tee ./logs/train/llama_7b_layerwise_condense_sparse_exhausive_${TASK}_bs_${bs}_sr${sr}_trial_${trial}_min_fro.log
