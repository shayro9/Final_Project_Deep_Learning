#!/bin/bash
#SBATCH --job-name=autoencoder_training   # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --mem=16GB                        # Total memory
#SBATCH --time=02:00:00                   # Time limit hrs:min:sec
#SBATCH --output=output_%j.log            # Standard output and error log

# Load necessary modules or activate environments
module load python/3.8
source activate final_project

# Run your main application
srun python main.py