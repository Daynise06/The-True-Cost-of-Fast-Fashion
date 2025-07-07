![](UTA-DataScience-Logo.png)

# The True Cost of Fast Fashion Impact

* **One Sentence Summary** This repository holds an attempt to build a regression model that predicts sustainability scores of fashion items using environmental and labor-impact features from the True Cost of Fast Fashion Impact Kaggle dataset.

## Overview

* This section could contain a short paragraph which include the following:
  * **Definition of the tasks / challenge**  The challenge is to use item-level fashion data to predict a sustainability score based on factors like CO₂ emissions, water usage, waste generation, and labor practices. The dataset helps quantify environmental and ethical impacts of clothing products.
  * **Your approach** The approach formulates the problem as a supervised regression task. We trained and compared multiple models including Linear Regression, Random Forest, and XGBoost to predict sustainability scores using environmental impact data and product metadata.
  * **Summary of the performance achieved** Our best-performing model (XGBoost) achieved an R² of 0.82, with MAE and RMSE significantly lower than baseline linear models, suggesting strong potential for identifying high- and low-sustainability items.


## Summary of Workdone

### Data

* Data:
  * Type: Input: CSV file with environmental and production-related features (e.g., emissions, water, price, labor indicators)
  * Output: Numeric sustainability score (continuous variable)
    
* Size:
  * Total size: ~300 KB
  * ~1,000 rows (fashion items)
    
* Instances (Train, Test, Validation Split):
  * 70% training
  * 15% validation
  * 15% test

#### Preprocessing / Clean up

* Removed duplicates and rows with missing sustainability scores
* Converted categorical features (e.g., brand, product type) using one-hot encoding
* Scaled numeric features (CO₂, water, waste, price) using StandardScaler
* Checked for outliers using z-scores

#### Data Visualization
* Histograms showed skewed distributions in CO₂ and water usage
* Boxplots showed brand-level differences in sustainability scores



### Problem Formulation

* Input: CO₂ emissions, water usage, waste, price, brand (encoded), labor index, product type
* Output:Sustainability Score (continuous, numeric)

Models:
* LinearRegression – for baseline
* RandomForestRegressor – handles non-linear patterns and feature importance
* XGBoostRegressor – boosted trees for better generalization and accuracy

Loss / Optimizer / Hyperparameters:
* Loss: MSE
* Grid search and random search used to optimize tree depth, number of estimators, and learning rate

### Training

* Python 3.10
* Jupyter Notebook
* Scikit-learn, XGBoost, matplotlib

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* XGBoost outperformed other models in both accuracy and interpretability
* Environmental features (CO₂ and water) were the most influential predictors
* Labor index contributed less to the score than expected
* Price was moderately correlated with higher sustainability

### Future Work

* Add more country-level or industry-level features
* Build a dashboard or recommendation tool to highlight sustainable brands
* Expand model for real-time use by consumers or sustainable fashion platforms

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Clone the repo or open in Google Colab
   * Run preprocess.ipynb to clean and format the dataset
   * Execute training-models.ipynb to train and save models
   * Run performance.ipynb for metrics and visualizations


### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: Helper functions for scaling, encoding, splitting
  * preprocess.ipynb: Cleans and transforms the raw CSV file
  * visualization.ipynb: Creates EDA charts (histograms, scatterplots, boxplots)
  * models.py: Defines regression models and tuning functions
  * training-model-1.ipynb: Linear Regression
  * training-model-2.ipynb: Random Forest
  * training-model-3.ipynb: XGBoost
  * performance.ipynb: Compares model metrics and plots results
  * final_report.ipynb: Combines analysis for export


### Software Setup
* List all of the required packages.
* pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

### Data

* Download from Kaggle: The True Cost of Fast Fashion
* Save the CSV as fast_fashion.csv in your working directory
* Run preprocess.ipynb to clean and prepare the dataset

### Training

* Run the corresponding training notebooks
* Models are saved as .pkl files in the /models folder

#### Performance Evaluation

* Use performance.ipynb to load models and generate evaluation metrics + plots


## Citations

* Dataset: Banerjee, Sourav. The True Cost of Fast Fashion, Kaggle. (https://www.kaggle.com/datasets/khushikyad001/the-true-cost-of-fast-fashion-impact)


