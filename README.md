# Credit Risk Prediction Using Machine Learning

A practical machine learning project for predicting loan default risk using borrower characteristics, loan attributes, and macroeconomic indicators. The workflow covers data cleaning, feature engineering, model comparison, evaluation, and serialization of an XGBoost-based prediction pipeline.

## Overview

Credit risk assessment helps financial institutions identify applications that may require closer review or additional screening. This project builds and evaluates several binary classification models using a loan dataset enriched with country-level economic factors.

The analysis compares four classifiers:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The models are assessed using accuracy, precision, recall, F1-score, and ROC-AUC, with a focus on correctly identifying positive-class cases.

The best-performing model in this study is XGBoost, which achieved an F1-score of 0.8871 and a ROC-AUC of 0.9941 on the held-out test set.

## Objectives

- Clean and standardize loan and macroeconomic datasets
- Engineer meaningful financial and temporal features
- Merge loan-level records with relevant economic indicators
- Build reusable preprocessing and modeling pipelines
- Compare multiple classification algorithms consistently
- Visualize performance using confusion matrices and ROC curves
- Save the selected pipeline for downstream inference

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib and Seaborn
- Joblib
- Jupyter Notebook

## Project Structure

```text
credit-risk-project/
├── 01_data_loading.ipynb
├── 02_data_cleaning.ipynb
├── 03_eda.ipynb
├── app.py
├── economic_indicators.csv
├── README.md
├── train.csv
├── train after tuning.ipynb
├── train before tuning.ipynb
├── xgboost_pipeline.joblib
└── .gitignore
```

## Data Processing and Feature Engineering

The project combines loan application records with annual macroeconomic data. The preprocessing work includes:

- standardizing column names and value formats
- converting date fields to datetime values
- reshaping economic indicators into analysis-ready features
- merging loan records with country-level indicators using country and disbursement year
- creating financial and temporal features such as:
  - loan disbursement year and month
  - interest amount
  - repayment ratio
  - log-transformed loan features
  - inflation, unemployment, exchange rate, and other macroeconomic signals

A scikit-learn `ColumnTransformer` is used to apply consistent transformations for both training and inference:

- Numerical features: median imputation + standardization
- Categorical features: most-frequent imputation + one-hot encoding
- Unknown categories are handled using `OneHotEncoder(handle_unknown="ignore")`

## Model Development

The following classifiers were trained and evaluated:

- Logistic Regression — baseline linear model with balanced class weights
- Decision Tree — interpretable tree-based classifier
- Random Forest — ensemble of 300 trees with balanced class weights
- XGBoost — boosted tree model tuned for strong classification performance

The dataset was split using stratified sampling:

- Training: 60%
- Validation: 20%
- Test: 20%

The held-out test set was used for final model comparison.

## Model Evaluation

Results on the held-out test set:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.9959 | 0.9016 | 0.8730 | 0.8871 | 0.9941 |
| Random Forest | 0.9933 | 0.7484 | 0.9563 | 0.8397 | 0.9922 |
| Decision Tree | 0.9940 | 0.9333 | 0.7222 | 0.8143 | 0.9905 |
| Logistic Regression | 0.9469 | 0.2539 | 0.9762 | 0.4029 | 0.9845 |

### Key findings

- XGBoost offers the best overall balance between precision and recall.
- Random Forest captures the most positive cases but produces more false positives.
- Decision Tree achieves the highest precision but misses more risky applicants.
- Logistic Regression has very high recall but weak precision, which limits its practical usefulness for this objective.

Confusion matrices and ROC curves were used to compare classification errors and discrimination performance across models.

## Model Artifact

The selected XGBoost pipeline was serialized using Joblib alongside the feature lists.

```python
joblib.dump(
    {
        "model": xgboost,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
    },
    "xgboost_pipeline.joblib",
)
```

This artifact can be reused for future scoring once the same preprocessing and feature-engineering logic are applied to new loan records.

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd credit-risk-project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the datasets

Place the project datasets in the working directory or configured data folder:

- `train.csv`
- `economic_indicators.csv`

### 5. Run the notebooks

Execute the notebooks in order from data loading through cleaning, EDA, feature engineering, model training, and evaluation.

## Future Improvements

- validate the target definition to reduce leakage risk
- evaluate with temporal validation on future loan applications
- tune decision thresholds for real-world business trade-offs
- add probability calibration and explainability tools
- introduce automated data validation and model versioning
- expand the project into a prediction API or deployment-ready service
- monitor performance and drift over time

## Responsible Use

This project is a machine learning prototype and should not be treated as a final lending decision system. Model predictions should be used as a supporting signal rather than the sole basis for approving or rejecting a loan application. Responsible deployment would require fairness testing, validation, transparency, and compliance with financial regulations.

## Author

Brian Mokua Oyaro

- GitHub: [Brian Oyaro](https://github.com/Brianoyaro)
- Portfolio: [Brian Mokua Oyaro's Portfolio](https://brianoyaro.github.io/Portfolio/)

## Connect

For project updates, data science work, and related portfolio work, you can explore the links above.
