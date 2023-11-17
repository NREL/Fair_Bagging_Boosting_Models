#!/bin/bash
#SBATCH --account=bias
#SBATCH --time 4:00:00
#SBATCH --nodes=9
#SBATCH --mail-user=erik.bensen@nrel.gov
#SBATCH --mail-type=ALL

module purge
module load conda
conda init bash

cd /projects/mobility/ebensen/Tree_Based_Model_Bias/

conda activate ai_bias

# Create array of model types
models=( rf gbt xgb )
corrections=( kendall distance pearson )

# Loop through models and corrections
for model in "${models[@]}"
do
    for correction in "${corrections[@]}"
    do
        srun -N 1 --exclusive python -W ignore gamma_sweeps.py --model_type $model --correction $correction
        # Sleep for 30 seconds to avoid data loading issues
        sleep 10
    done
done
wait