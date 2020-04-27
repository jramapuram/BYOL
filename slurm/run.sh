#!/bin/bash -l

#SBATCH --job-name=SimCLR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-15                                                  # creates 16 tasks
#SBATCH --partition=<MY_PARTITIONS_HERE>                              # Fill this in!
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1                                                  # 1 GPU per process
#SBATCH --mem=16000                                                   # Memory per GPU
#SBATCH --constraint="COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"  # Use newer GPUs, remove if not needed

echo $CUDA_VISIBLE_DEVICES
module load GCCcore/8.2.0 Singularity/3.4.0-Go-1.12 CUDA/10.1.243     # load corresponding modules based on your cluster

MASTER=$(squeue | grep ${SLURM_ARRAY_JOB_ID}_0 | awk {'print $8'})    # job0 is the master
echo "master is $MASTER"

# Modify this command as appropriate; pytorch1.5.0_cuda10.1.simg is a singularity image of
# the following docker image: jramapuram/pytorch:1.5.0-cuda10.1 with the addition that /opt/conda is chdmod'd.
# This can be done as follows:

# Create a file called MYRULES which contains:

# Bootstrap: docker
# From: jramapuram/pytorch:1.5.0-cuda10.1
# %post
#     mkdir -p /opt
#     chmod -R 777 /opt

# Then build the container:
# sudo singularity build pytorch1.5.0_cuda10.1.simg MYRULES

# Profit (after setting DATA_DIR below)

srun --unbuffered singularity exec -B <DATA_DIR>:/datasets \
     --nv $HOME/docker/pytorch1.5.0_cuda10.1.simg python ../main.py \
     --epochs=100 \
     --data-dir=/datasets \
     --batch-size=1024 \
     --convert-to-sync-bn \
     --visdom-url=http://MY_VISDOM_URL \
     --visdom-port=8097 \
     --num-replicas=16 \
     --distributed-master=$MASTER \
     --distributed-port=29301 \
     --distributed-rank=${SLURM_ARRAY_TASK_ID} \
     --uid=simclrv00_0
