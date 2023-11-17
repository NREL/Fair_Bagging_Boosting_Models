# Tree Based Model Bias
Contains the class to build a gradient boosted trees and random forest  model with XGBoost and test their respective biases.

## Instructions
Optimizing the base models
* Copy base_model_opt.py and run_base_opt.slurm to eagle and submit the slurm batch file
* This should generate a folder called base_models that has subfolders for each model type
* Copy the base_model folder to the local computer and then you can run the notebook base_model_bias_testing.ipynb to test bias and save plots
* Note: this all should already be done so you shouldn't need to redo this

Optimization Bound Experiments
* These are the experiments that create the results_{number} folders
* You can edit the model, correction and leeway lists in run_bias_pipeline.slurm to adjust what combinations are run
* To run, copy bias_pipeline.py and run_bias_pipeline.slurm to eagle then submit the slurm batch file
* Copy the resulting results_{number} folders locally to within the results folder of this repo
* The bias_pipeline.py does not correctly test bias or plot things on eagle for some reason, so either run the desired combinations individually using investigate_bias.ipynb, or in bulk by editing the lists in lines 179-181 of process_bias.py and running the batch script locally.
* After the processing is done, you can generate the paper figures using results_trends.ipynb located within the results folder

Gamma Sweep Experiments
* These are the experiments that create the results_{model_type} folders
* You can edit the model types and corrections in run_gamma_sweeps.slurm to adjust which combinations are run
* To run, copy gamma_sweeps.py and run_gamma_sweeps.slurm to eagle and submit the slurm batch file
* Copy the resulting results_{model_type} folders to the local machine within the results folder of this repo
* Then you can generate the paper figures by running results_trends_gamma.ipynb

Note: the files are already on eagle under "/projects/mobility/ebensen/Tree_Based_Model_Bias/"