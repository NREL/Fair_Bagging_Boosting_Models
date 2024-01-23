# Fair Tree-based Ensemble Regression Models
This repository contains a framework for building, measuring bias and correcting bias in gradient boosted trees (GBT), random forest (RF), and XGBoost  models using the XGBoost library. This takes advantage of the flexibility in XGBoost library to represent gradient boosted tree and random forest models, as well as the ability to use custom loss function.

## Testing for Bias
The [base_model_bias_testing.ipynb](https://github.com/NREL/Fair_Forest_Models/blob/main/base_model_bias_testing.ipynb) notebook is used to test for bias in a pre-trained XGBoost, GBT, and RF models. This approach can also be used to test bias in other machine learning models that can be implemented in this framework. The bias testing method takes as input:
* Pre-trained machine learning model.
* Test dataset to calculate the model's error.

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
