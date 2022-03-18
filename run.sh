#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000
#SBATCH --gres=gpu:2
#SBATCH --exclude=lme170,lme171,lme221,lme53
#SBATCH -o /cluster/[my_name]/logs/%x-%j-on-HERE-%N.out
#SBATCH -e /cluster/[my_name]/logs/%x-%j-on-HERE-%N.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds"
#SBATCH --time=24:00:00

export WORKON_HOME==/cluster/[my_name]/.python_cache

pip3 install --user -r requirements.txt

python3 train.py --cluster --batch_size 8 --lr 0.01 --optim sdg --scheduler none 
