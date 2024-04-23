# Sapling
### Set up Environment
Create a conda environment: ``conda create -n trimllama python=3.10 -y``

Load the environment: ``conda activate trimllama``

Install all the requirements from lm_eval: ``cd lm_eval/lm-evaluation-harness`` then ``pip install -e .``

Go to scripts in the main directory ``cd ../scripts``

Things to install if no requirements:
datasets
matplotlib
evaluate
sentencepiece

Since we are using transformers=4.31.0, I cloned PEFT=0.4.0

bash run_decomposition.sh 'sciq' 'TheBloke/Llama-2-7B-fp16' 3 10

### STEP 1: Model compression for domain-specific LLMs — TrimLLaMA

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

### STEP 2: Evaluation.

build `lm_eval` locally, this is modified from [this repo](https://github.com/EleutherAI/lm-evaluation-harness).
```bash
cd lm_eval/lm-evaluation-harness
pip install -e .
```
