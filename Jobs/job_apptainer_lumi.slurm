#!/bin/bash -l
#SBATCH --job-name=lumi
#SBATCH --account=project_4650000xx
#SBATCH --time=00:10:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --output=%x-%j.out
#SBATCH --exclusive

N=$SLURM_JOB_NUM_NODES
echo "--nbr of nodes:", $N
echo "--total nbr of gpus:", $SLURM_NTASKS

MyDir=/project/project_4650000xx
MyApplication=${MyDir}/FiniteVolumeGPU_HIP/mpiTesting.py
Container=${MyDir}/FiniteVolumeGPU_HIP/my_container.sif

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

export MPICH_GPU_SUPPORT_ENABLED=1

srun --cpu-bind=${CPU_BIND} --mpi=pmi2 \
     apptainer exec "${Container}" \
     python ${MyApplication} -nx 1024 -ny 1024 --profile
