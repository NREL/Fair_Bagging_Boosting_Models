#!/bin/bash
#SBATCH --account=bias
#SBATCH --time 8:00:00
#SBATCH --nodes=3
#SBATCH --mail-user=erik.bensen@nrel.gov
#SBATCH --mail-type=ALL

module purge
module load conda
conda init bash

cd /projects/mobility/ebensen/Tree_Based_Model_Bias/

conda activate ai_bias

# Create array of model types
models=( rf gbt xgb )

# Loop through models and corrections
for model in "${models[@]}"
do
    srun -N 1 --exclusive python -W ignore base_model_opt.py --model_type $model &
    # Sleep for 30 seconds to avoid data loading issues
    sleep 10
done
wait