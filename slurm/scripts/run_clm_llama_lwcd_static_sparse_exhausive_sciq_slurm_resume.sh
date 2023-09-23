#!/bin/bash
#SBATCH --job-name=sapling_llama_7b_sciq_exhausive # create a short name for your job
#SBATCH --nodes=1
#SBATCH --gres=gpu:16     # number of gpus per node
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=30-00:00:00     # total run time limit (HH:MM:SS)
#SBATCH --reservation=high-profile
#SBATCH --partition=high-profile
#SBATCH --error=/nfs/projects/mbzuai/ext_hao.zhang/lanxiang/Sapling/slurm/logs/sbatch/%J.%N.exhausive_sciq.err
#SBATCH --output=/nfs/projects/mbzuai/ext_hao.zhang/lanxiang/Sapling/slurm/logs/sbatch/%J.%N.exhausive_sciq.out

export SLURM_JOB_NODELIST=p4-r66-a.g42cloud.net
export SLURM_JOB_NUM_NODES=1
export SLURM_NTASKS_PER_NODE=1

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= " $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
# If you want to load things from your .bashrc profile, e.g. cuda drivers, singularity etc
cd /nfs/projects/mbzuai/ext_hao.zhang/lanxiang
source .bashrc
conda activate lanxiang_llm
cd /nfs/projects/mbzuai/ext_hao.zhang/lanxiang/Sapling/slurm/scripts
free -g 2>&1
lscpu 2>&1
# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work
export MASTER_PORT=20005
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
#echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR"=$MASTER_ADDR
#echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************
echo "Run started at:- "
date
# Actual run of script
#srun python main.py # Use this if you have python in your environment
srun run_clm_llama_lwcd_static_sparse_exhausive_resume.sh sciq decapoda-research/llama-7b-hf 2 0.625 [1,4,5,13,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31] 4