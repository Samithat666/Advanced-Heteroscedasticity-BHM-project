# Advanced Heteroscedasticity Modeling with Bayesian Hierarchical Regression


## Project Overview

This project demonstrates advanced regression modeling by explicitly modeling **heteroscedasticity (non-constant variance)** using a **Bayesian Hierarchical Model (BHM)**. The predictive performance and uncertainty quantification of the Bayesian model are compared against **Ordinary Least Squares (OLS)** with heteroscedasticity-robust standard errors (HC3).

The main objective is to show that **explicit variance modeling improves prediction accuracy and uncertainty estimation**.

---

## Objectives

* Generate synthetic data with **structured heteroscedasticity**
* Implement a **two-part Bayesian Hierarchical Model**
* Compare against **OLS with HC3 robust standard errors**
* Evaluate **out-of-sample predictive performance (RMSE)**
* Perform **MCMC diagnostics**
* Conduct **posterior predictive checks**
* Interpret **variance parameters**

---

## Synthetic Data Generation

A dataset of size **N = 1000** is programmatically generated using NumPy.

**Data Generating Process**

[
y_i = \beta_0 + \beta_1 x_i + \epsilon_i
]

Error variance follows:

[
\sigma_i = \exp(\alpha_0 + \alpha_1 x_i)
]

This induces **strong heteroscedasticity**, where variance increases with the predictor variable.

---

## Models

### 1. OLS with HC3 Robust Errors

* Implemented using `statsmodels`
* Uses heteroscedasticity-robust covariance (HC3)
* Does **not explicitly model variance**

---

### 2. Bayesian Hierarchical Model (BHM)

#### Mean Model

[
\mu_i = \beta_0 + \beta_1 x_i
]

#### Variance Model

[
\log(\sigma_i) = \alpha_0 + \alpha_1 x_i
]

#### Likelihood

[
y_i \sim \mathcal{N}(\mu_i, \sigma_i)
]

#### Priors

* β₀, β₁ ~ Normal(0, 10)
* α₀, α₁ ~ Normal(0, 1)

This structure explicitly models how **variance changes with predictor**.

---

## Model Evaluation

Models are compared using **test RMSE** on held-out data.

| Model                       | Test RMSE              |
| --------------------------- | ---------------------- |
| OLS (HC3)                   | See `outputs/rmse.txt` |
| Bayesian Hierarchical Model | See `outputs/rmse.txt` |

The Bayesian model typically provides **better predictive accuracy**.

---

## Variance Interpretation

Variance equation:

[
\log(\sigma_i) = \alpha_0 + \alpha_1 x_i
]

* α₁ > 0 → Variance increases with predictor → Heteroscedasticity confirmed
* α₁ ≈ 0 → Homoscedastic
* Posterior of α₁ quantifies uncertainty in variance

---

## Diagnostics & Validation

### MCMC Diagnostics

* R-hat ≈ 1 indicates convergence
* Effective Sample Size (ESS)
* Trace plots for posterior sampling

Saved in:

* `outputs/mcmc_diagnostics.csv`
* `outputs/traceplot.png`

### Posterior Predictive Checks

* Validates model fit to data

Saved in:

* `outputs/posterior_predictive.png`

### Residual Analysis (OLS)

Shows heteroscedastic pattern:

Saved in:

* `outputs/ols_residuals.png`

---

## Key Comparison

| Feature                    | OLS (HC3)     | Bayesian Hierarchical   |
| -------------------------- | ------------- | ----------------------- |
| Handles heteroscedasticity | Adjusts SE    | Explicit variance model |
| Predictive accuracy        | Moderate      | Better                  |
| Variance interpretation    | Not available | Direct                  |
| Uncertainty quantification | Limited       | Full posterior          |
| Model flexibility          | Low           | High                    |



---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to run

```bash
python bhm_project.py
```


## Conclusion

The Bayesian Hierarchical Model successfully captures structured heteroscedasticity and provides improved predictive performance and superior uncertainty quantification compared to OLS with robust errors. Explicit variance modeling is essential when error variance depends on predictors, and Bayesian hierarchical methods provide a powerful solution.

