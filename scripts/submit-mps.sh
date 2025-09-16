#!/bin/bash
#SBATCH --time=0-1:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=2
#SBATCH --gpus=3g.20gb:1
#SBATCH --account=def-sponsor00

mkdir -p $HOME/tmp
export CUDA_MPS_LOG_DIRECTORY=$HOME/tmp
nvidia-cuda-mps-control -d

rm output*
export NUM_CPUS=$2
module load python/3.11
for ((i=0; i<NUM_CPUS; i++))
 do
 echo $i
 python matmul-mps.py $1 > output$i   &
 done

wait
