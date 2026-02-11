# Advanced Heteroscedasticity Modeling with Bayesian Hierarchical Regression

import numpy as np
import matplotlib.pyplot as plt
import os
import pymc as pm
import arviz as az
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    # -------------------------------------------------
    # Setup
    # -------------------------------------------------
    np.random.seed(42)
    os.makedirs("outputs", exist_ok=True)

    
    # 1. Generate Synthetic Heteroscedastic Data
    
    N = 1000
    x = np.random.uniform(0, 5, N)

    beta0_true = 2.0
    beta1_true = 1.5
    alpha0_true = -0.5
    alpha1_true = 0.4

    sigma_true = np.exp(alpha0_true + alpha1_true * x)
    y = beta0_true + beta1_true * x + np.random.normal(0, sigma_true)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # -------------------------------------------------
    # 2. OLS Baseline with HC3 Robust Errors
    # -------------------------------------------------
    X_train_ols = sm.add_constant(x_train)
    X_test_ols = sm.add_constant(x_test)

    ols_model = sm.OLS(y_train, X_train_ols).fit(cov_type="HC3")
    ols_pred = ols_model.predict(X_test_ols)
    ols_rmse = np.sqrt(mean_squared_error(y_test, ols_pred))

    # Residual plot
    residuals = y_train - ols_model.predict(X_train_ols)
    plt.figure()
    plt.scatter(x_train, residuals, alpha=0.5)
    plt.xlabel("Predictor")
    plt.ylabel("Residuals")
    plt.title("OLS Residuals vs Predictor")
    plt.savefig("outputs/ols_residuals.png")
    plt.close()

    
    # 3. Bayesian Hierarchical Model (PyMC 5 Safe)
    
    with pm.Model() as bhm:

        # Mutable data container
        x_shared = pm.Data("x_shared", x_train)

        # Mean model priors
        beta0 = pm.Normal("beta0", mu=0, sigma=10)
        beta1 = pm.Normal("beta1", mu=0, sigma=10)

        # Variance model priors
        alpha0 = pm.Normal("alpha0", mu=0, sigma=1)
        alpha1 = pm.Normal("alpha1", mu=0, sigma=1)

        # Mean and variance equations
        mu = beta0 + beta1 * x_shared
        log_sigma = alpha0 + alpha1 * x_shared
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

        # Likelihood (TRAINING ONLY — fixed size)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        # Prediction variable (KEY FIX for shape mismatch)
        y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma)

        # Sampling
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=2,
            target_accept=0.9,
            return_inferencedata=True
        )

   
    # 4. Posterior Predictive Check (TRAIN)
    
    with bhm:
        ppc_train = pm.sample_posterior_predictive(trace, var_names=["y_pred"])

    bhm_train_pred = ppc_train.posterior_predictive["y_pred"].mean(
        dim=("chain", "draw")
    ).values

    plt.figure()
    plt.scatter(y_train, bhm_train_pred, alpha=0.5)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("Posterior Predictive Check (Training)")
    plt.savefig("outputs/posterior_predictive_train.png")
    plt.close()

    # 5. Posterior Predictive (TEST — SAFE)
    
    with bhm:
        pm.set_data({"x_shared": x_test})
        ppc_test = pm.sample_posterior_predictive(trace, var_names=["y_pred"])

    bhm_test_pred = ppc_test.posterior_predictive["y_pred"].mean(
        dim=("chain", "draw")
    ).values

    bhm_rmse = np.sqrt(mean_squared_error(y_test, bhm_test_pred))


    # 6. MCMC Diagnostics
    
    summary = az.summary(trace)
    summary.to_csv("outputs/mcmc_diagnostics.csv")

    az.plot_trace(trace)
    plt.savefig("outputs/traceplot.png")
    plt.close()


    # 7. Save RMSE Results
    
    with open("outputs/rmse.txt", "w") as f:
        f.write(f"OLS RMSE (test): {ols_rmse:.4f}\n")
        f.write(f"BHM RMSE (test): {bhm_rmse:.4f}\n")

    print("===================================================")
    print("Project completed successfully.")
    print(f"OLS RMSE (test): {ols_rmse:.4f}")
    print(f"BHM RMSE (test): {bhm_rmse:.4f}")
    print("Check 'outputs/' folder for diagnostics and plots.")
    print("===================================================")


# Windows multiprocessing safety guard

if __name__ == "__main__":
    main()
