#!/bin/bash

#------- Descripción del trabajo -------

#SBATCH --job-name='hello_world'
#SBATCH --partition=gpu

#------- Avisos -------

#SBATCH --mail-type=all
#SBATCH --mail-user=angel.berihuete@uca.es

#------- Parametrización -------

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:02:00

#------- Entrada/Salida -------

#SBATCH --output=hello_%j.out
#SBATCH --error=hello_%j.err

#------- Carga de módulos -------

module purge
module load tensorflow-gpu

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Job NPROCS: ${SLURM_NPROCS}"
echo "== Job NNODES: ${SLURM_NNODES}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"

#------- Comando -------
#- export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python test_edl.py
