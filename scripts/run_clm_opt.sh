TASK=$1               # "hellaswag"
MODEL=$2              # "facebook/opt-1.3b"
bs=$3                 # 8
lr=$4                 # default: 2e-5

export WANDB_DISABLED=true

### baseline ft
torchrun --nproc_per_node=8 \
    --master_port 20680 \
    run_clm.py \
    --bf16 True \
    --model_name_or_path $MODEL \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs 3 \
    --seed 42 \
    --do_train \
    --do_eval \
    --save_steps 20000 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'OPTDecoderLayer' \
    --output_dir outputs/$MODEL/${TASK}_bs_${bs}/baseline_with_eval_${lr} 2>&1 | tee ./logs/train/opt-1.3b_baseline_${TASK}_bs_${bs}_with_eval_$lr.log