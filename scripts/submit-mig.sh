#!/bin/bash
#SBATCH --time=0-1:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=2
#SBATCH --gpus=3g.20gb:1
#SBATCH --account=def-sponsor00
python ./matmul-mig.py > output


