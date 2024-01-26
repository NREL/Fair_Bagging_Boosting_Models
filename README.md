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
* To run gamma_sweeps.py in terminal:
### Script Parameters
Model Type
| Parameter | Description |
| ----------- | ----------- |
| rf | Random Forest Model|
| gbt | Gradient Boosted Trees |
| xgb | Extreme Gradient Boosted Trees |

Correction Terms
| Parameter | Description |
| ----------- | ----------- |
| pearson | Pearson correlation for linear relationship|
| distance | Distance correlation for non-linear |
| kendall | Kendall's Tau for non-linear. This will run the slowest |

```linux
python -W ignore gamma_sweeps.py --model_type {model type} --correction {correction term}
```
should look like this is running proper:
```linux
Model: rf, Correction: distance, Loading data...
Model: rf, Correction: distance, Sweeping Gamma...
Model: rf, Correction: distance, Gamma: 0.5, 1/51
[10:44:59] WARNING: /var/folders/nz/j6p8yfhx1mv_0grj5xl4650h0000gp/T/abs_21wtzqx5vy/croot/xgboost-split_1675457780668/work/src/learner.cc:767: 
Parameters: { "num_parallel_trees" } are not used.


```
* Results will populate in  the results_{model_type} folders to the local machine within the results folder of this repo
* Then you can generate the paper figures by running results_trends_gamma.ipynb

