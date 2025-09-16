mkdir -p $HOME/tmp
export CUDA_MPS_LOG_DIRECTORY=$HOME/tmp
nvidia-cuda-mps-control -d

rm output*
export NUM_CPUS=$2
module load python/3.11
for ((i=0; i<NUM_CPUS; i++))
 do
 echo $i
 python matmul-mps.py $1 > output$i  &
 done
wait
