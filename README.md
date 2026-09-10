# Medical Insurance Cost Prediction

An end-to-end machine learning regression project that predicts medical insurance
**charges** from a person's demographic and lifestyle information, deployed as an
interactive Streamlit app.

## Objective

Build and deploy a Linear Regression model that estimates medical insurance charges
based on age, sex, BMI, number of children, smoking status, and region.

## Dataset

[Medical Insurance Cost Prediction dataset](https://www.kaggle.com/datasets/hetmengar/medical-insurance-cost-prediction) (Kaggle), 1,338 records.

| Column | Type | Description |
|---|---|---|
| age | Numerical | Age of the beneficiary |
| sex | Categorical | Sex |
| bmi | Numerical | Body Mass Index |
| children | Numerical | Number of children/dependents |
| smoker | Categorical | Smoking status |
| region | Categorical | Residential region |
| charges | Numerical | **Target** \u2014 medical insurance charges |

## Features and Target

- **X:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **y:** `charges`

## EDA Findings

- `charges` is right-skewed, with a long tail of high-cost cases.
- **Smoking status is by far the strongest predictor** \u2014 smokers' charges are dramatically higher than non-smokers', with almost no overlap.
- Age has a positive, fairly linear relationship with charges.
- BMI matters much more for smokers than non-smokers \u2014 a real interaction effect.
- Region and number of children show little relationship with charges.

## Preprocessing

- Numerical features (`age`, `bmi`, `children`) passed through unchanged.
- Categorical features (`sex`, `smoker`, `region`) One-Hot Encoded (`drop="first"`).
- Implemented with a scikit-learn `ColumnTransformer` inside a `Pipeline`, fitted only on the training split to avoid data leakage.

## Model

**Linear Regression**, trained in two versions:
1. **Baseline** \u2014 raw features only.
2. **Feature-engineered (final)** \u2014 adds `age \u00d7 smoker` and `bmi \u00d7 smoker` interaction terms, motivated by the EDA and by a non-random pattern in the baseline's residual plot.

## Evaluation Metrics

| Model | R\u00b2 | Adjusted R\u00b2 | MAE | MSE | RMSE |
|---|---|---|---|---|---|
| Baseline | 0.807 | 0.802 | $4,177 | 35,478,021 | $5,956 |
| **Feature-Engineered (final)** | **0.886** | **0.882** | **$2,832** | **20,956,066** | **$4,578** |

The feature-engineered model was selected because it improves **every** metric consistently, not just R\u00b2 in isolation.

## Model Improvement Experiment

Added `age\u00d7smoker` and `bmi\u00d7smoker` interaction features after EDA and residual analysis suggested the baseline model wasn't capturing how much more steeply BMI/age drive costs specifically among smokers. This dropped RMSE by ~23%.

## Residual Analysis

The baseline model's residuals showed a fan-shaped pattern and a cluster of large under-predictions at higher charge levels, mostly for high-BMI smokers \u2014 the motivation for the interaction-feature experiment above.

## How to Run Locally

```bash
git clone <this-repo-url>
cd medical-insurance-prediction
pip install -r requirements.txt
streamlit run app.py
```

To reproduce training from scratch, open and run `notebooks/medical_insurance_analysis.ipynb` top to bottom \u2014 it regenerates `models/insurance_model.pkl`.

## Deployed App

_Add your Streamlit Community Cloud URL here after deploying (see below)._

## Repository Structure

```
medical-insurance-prediction/
\u251c\u2500\u2500 data/
\u2502   \u2514\u2500\u2500 medical-charges.csv
\u251c\u2500\u2500 models/
\u2502   \u251c\u2500\u2500 insurance_model.pkl
\u2502   \u2514\u2500\u2500 model_metadata.pkl
\u251c\u2500\u2500 notebooks/
\u2502   \u251c\u2500\u2500 medical_insurance_analysis.ipynb
\u2502   \u2514\u2500\u2500 plots/
\u251c\u2500\u2500 pipeline_utils.py
\u251c\u2500\u2500 app.py
\u251c\u2500\u2500 requirements.txt
\u251c\u2500\u2500 README.md
\u2514\u2500\u2500 .gitignore
```

## Limitations & Responsible Use

- Trained on a small (1,337-row) sample dataset \u2014 not representative of any real insurer's actual pricing.
- Linear Regression assumes linear relationships; real premium pricing involves many more factors and non-linear rules.
- **This model must not be used for real medical or insurance underwriting decisions.** It is an educational project only.
- Correlation shown in the EDA does not imply causation.
