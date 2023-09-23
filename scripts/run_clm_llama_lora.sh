TASK=$1               # "hellaswag"
MODEL=$2              # "decapoda-research/llama-7b-hf"
bs=$3                 # 8
ep=$4                 # 17, number of training epochs
lora_r=$5             # 8
lora_alpha=$6         # 16

export WANDB_DISABLED=true
export TORCH_DISTRIBUTED_DEBUG=DETAIL

### lora ft
WORLD_SIZE=8 torchrun --nproc_per_node=8 \
    --master_port 20679 \
    run_clm_lora.py \
    --fp16 True \
    --model_name_or_path $MODEL \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate 2e-5 \
    --num_train_epochs $ep \
    --torch_dtype float16 \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout 0.05 \
    --seed 42 \
    --do_train \
    --do_eval \
    --save_steps 20000 \
    --output_dir outputs/$MODEL/${TASK}_bs_${bs}/lora_ep_${ep}_r_${lora_r}_alpha_${lora_alpha} 2>&1 | tee ./logs/train/llama_7b_lora_${TASK}_bs_${bs}_ep_${ep}_r_${lora_r}_alpha_${lora_alpha}.log