# Fair Forests Regression Models

## Contributors:
Eric A. Bensen: ebensen@andrew.cmu.edu  
Joseph Severino: joseph.severino@nrel.gov  
Juliette Ugirumurera: jugirumu@nrel.gov  

## Description
This repository contains a framework for building, measuring bias and correcting bias in 3 popular bagging and boosting regression models: gradient boosted trees (GBT), random forest (RF), and XGBoost  models, using the XGBoost library. The framework takes advantage of the flexibility in XGBoost library to represent gradient boosted tree and random forest models, as well as the ability to use custom loss function. This code is associated with NREL software record SWR-24-38.

## Setting up conda environment
```
conda env create -f environment.yml
conda activate ai_bias
```
To enable the ai_bias conda environment in Jupyter notebook:
```
conda activate ai_bais
conda install ipykernel
python -m ipykernel install --user --name=ai_bias
```
After these steps, you should see ai_bias as a kernel in the Jupyter notebook interface.

## Scripts Parameters
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

Demographic attribute
|CDC Variable Name  |  Description |
| ----------------- | -------------- |
| AGE17  |  Persons aged 17 and younger |
| AGE65 | Persons aged 65 and older   |
| CROWD | At household level (occupied housing units), more people than rooms   |
| DISABL  |  Civilian non-institutionalized population with a disability |
| GROUPQ | Persons in group quarters |
| LIMENG | Persons (age 5+) who speak English "less than well" |
| MINRTY | Minority (all persons except white, non-Hispanic) |
| MOBILE | Mobile Homes |
| MUNIT | Housing in structures with 10 or more units estimate |
| NOHSDP | Persons (age 25+) with no high school diploma |
| NOVEH | Households with no vehicle available |
| PCI | Per capita Income |
| POV | Persons below proverty |
| SNGPNT | Single parent household with children under 18  |
| UNEMP | Civilian (age 16+) unemployed |


## Train Baseline GBT, RF, and XGBoost
This steps allows to train baseline GBT, RF or XGBoost forest models to test bias on.
```
python base_model_opt.py --model_type {model type} --demographic {demographic attribute}
```
Use the table above to see allowed values for the --model_type and --demographic parameters. The minority demographic **MINRTY** is the default demographic value.

## Testing for Bias
The [base_model_bias_testing.ipynb](https://github.com/NREL/Fair_Forest_Models/blob/main/base_model_bias_testing.ipynb) notebook is used to test for bias in a pre-trained XGBoost, GBT, and RF models. This approach can also be used to test bias in other machine learning models that can be implemented in this framework. The bias testing method takes as input:
* Pre-trained machine learning model in the **base_models** folder.
* Test dataset to calculate the model's error. This repo includes some test files in the **data** folder.

## Bias Mitigation
To do bias mitigation, we train the tree-based models with a custom loss function that combines the training loss function and a correction term that penalizes high correlation between the model's error and a protected attribute. We use a parameter gamma, with values in [0,1], to understand the trade-off between model performance and demographic bias in the models. That is, when gamma = 0, the training process ignores the correction terms and focus on maximizing the model's accuracy. When gamma = 1, the training process only focuses on minimizing the model's bias. We use the gamma_sweeps.py to do a thorough analysis of the models types and the correction terms. See below for more instructions.

**Gamma:**
Gamma is tested with values: {0} and 50 values drawn from interval [0.5, 0.9999].
You can modify the set of gammas considered by changing line 144 and 146 in **gamma_sweeps.py** script.


### To Run gamma_sweeps.py in Terminal:
```linux
python -W ignore gamma_sweeps.py --model_type {model type} --correction {correction term} --demographic {demographic attribute}
```
The minority demographic (**MINRTY**) is the default protected demographic attribute if the **--demographic** is not used.

If the code is running properly, you should see the output like below:
```{r, message=TRUE}
Model: {model type}, Correction: distance, Loading data...
Model: {model type}, Correction: distance, Sweeping Gamma...
Model: {model type}, Correction: distance, Gamma: 0.5, 1/51
...

```
Results will populate in  the results_{model_type} folders within this repo's folder.

## Visualizing Results
To plot results please use this [visualization notebook](https://github.com/NREL/Fair_Forest_Models/blob/main/visualization_of_results.ipynb).
