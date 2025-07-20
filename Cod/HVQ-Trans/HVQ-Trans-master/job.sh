#!/bin/bash
#----------------------------------------------------
# Example SLURM job script with SBATCH
#----------------------------------------------------
#SBATCH -J myjob            # Job name
#SBATCH -o myjob_%j.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e myjob_%j.e       # Name of stderr output file(%j expands to jobId)
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 5:00:00         # Run time (hh:mm:ss)
#SBATCH --mem-per-cpu=3G    # Memory per core demandes ( 96GB = 3GB * 32 cores)
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1

module load cesga/2020
srun hostname
source venv/bin/activate
python3 -m hvq.tools.train_val --config experiments/oitaven/config.yaml
deactivate
echo "done"                 # Write this message on the output file when finished
