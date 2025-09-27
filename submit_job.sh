#!/usr/bin/bash

#SBATCH -J space
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g5
#SBATCH -t 1-0
#SBATCH -o ./logs/slurm-%A.out
#SBATCH -e ./logs/slurm-err-%A.out

python - << 'EOF'
import torch
print(torch.cuda.is_available())
EOF

python models/cpt.py --train_size 0.2