#!/bin/bash
#SBATCH --account=bias
#SBATCH --time 12:00:00
#SBATCH --nodes=36
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
leeways=( 0.1 0.15 0.25 1.0 )

# Loop through models and corrections
for leeway in "${leeways[@]}"
do
    for model in "${models[@]}"
    do
        for correction in "${corrections[@]}"
        do
            srun -N 1 --exclusive python -W ignore bias_pipeline.py --model_type $model --correction $correction --leeway $leeway &
            # Sleep for 30 seconds to avoid data loading issues
            sleep 10
        done
    done
done
wait