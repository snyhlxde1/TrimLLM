# Sapling

## Setup

install dependencies:

```bash
conda create -n sapling python==3.9
conda activate ssapling
pip install -r requirements.txt
```

patches:
```
cd Sapling
cp -rv trainer_pt_utils.py ~/anaconda3/envs/peft/lib/python3.9/site-packages/transformers/
cp -rv modeling_llama.py ~/anaconda3/envs/peft/lib/python3.9/site-packages/transformers/models/llama
cp -rv fully_sharded_data_parallel.py ~/anaconda3/envs/peft/lib/python3.9/site-packages/torch/distributed/fsdp/
```

## Usage

STEP 1: model compression for domain-specific LLMs — TrimLLaMA

run LLaMA full fine-tuning:
```bash
cd scripts
bash run_clm_llama.sh {task} {model_path} {batch_size} {lr}
```

run TrimLLaMA:
```bash
bash run_clm_llama_lwcd_static_sparse_exhausive.sh {task} {model_path} {batch_size} {lr} {trial_number}
```

other important arguments (optional):
```bash
--tie_breaker_strategy 'activation' or 'naive' # 'activation' # tie-breaker strategy for layer dropping: naive that drops the one in the front or activation-based that drops the one with max activation norm.
--sparsity_ratio 0.75 # for example, r = 0.75: ratio of frozen parameters vs. trainable parameters.
--max_budget: 48 # maximum number of MLP/attention modules that can be removed before exiting.
```

STEP 2: Evaluation.

build `lm_eval` locally, this is modified from [this repo](https://github.com/EleutherAI/lm-evaluation-harness).
```bash
cd lm_eval/lm-evaluation-harness
pip install -e .
```
