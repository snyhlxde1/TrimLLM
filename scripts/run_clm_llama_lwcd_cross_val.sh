TASK=$1               # "hellaswag"
MODEL=$2              # "decapoda-research/llama-7b-hf"
bs=$3                 # 8
trial=$4              # trial number: 1, 2, 3, ... etc.

export WANDB_DISABLED=true

### baseline ft
torchrun --nproc_per_node=8 \
    --master_port 20695 \
    run_clm_llama_lwcd_cross_val.py \
    --bf16 True \
    --model_name_or_path $MODEL \
    --total_layer_count 32 \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --seed 42 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --output_dir outputs/$MODEL/${TASK}_bs_${bs}/layerwise_condense_cross_eval_trial_$4 2>&1 | tee ./logs/train/llama_7b_layerwise_condense_${TASK}_bs_${bs}_cross_eval_trial_$4.log