import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import exoplanet as xo
from celerite2.theano import terms, GaussianProcess

def run_gp_model(time, flux, flux_err, tune=3000, draws=1000):
    # Normalize
    t = time - np.min(time)
    y = (flux - np.mean(flux)) / np.std(flux)
    yerr = flux_err / np.std(flux)

    with pm.Model() as model:
        # Long-term trend (squared exponential)
        log_sigma_trend = pm.Normal("log_sigma_trend", mu=np.log(np.std(y)), sigma=5)
        log_rho_trend = pm.Normal("log_rho_trend", mu=np.log((t.max()-t.min())/2), sigma=5)
        trend_kernel = terms.SHOTerm(log_sigma=log_sigma_trend, log_rho=log_rho_trend, Q=1.0)

        # Quasi-periodic variability
        log_amp = pm.Normal("log_amp", mu=np.log(np.std(y)), sigma=5)
        log_period = pm.Normal("log_period", mu=np.log(1.0), sigma=5)  # period ~exp(1 day)
        log_Q = pm.Normal("log_Q", mu=np.log(15), sigma=5)
        log_rho = pm.Normal("log_rho", mu=np.log(1.0), sigma=5)
        
        periodic_kernel = terms.SHOTerm(log_sigma=log_amp, log_rho=log_rho, Q=pm.Deterministic("Q", pm.math.exp(log_Q)))

        # Combined kernel
        kernel = trend_kernel + periodic_kernel
        gp = GaussianProcess(kernel, t, diag=yerr**2, mean=0.0)

        gp.compute()

        pm.Normal("obs", mu=gp.predict(), sigma=yerr, observed=y)

        trace = pm.sample(tune=tune, draws=draws, chains=2, target_accept=0.95, return_inferencedata=True)

    return model, trace

def plot_gp_fit(time, flux, flux_err, model, trace, n_samples=100):
    t = time - np.min(time)
    y = (flux - np.mean(flux)) / np.std(flux)
    yerr = flux_err / np.std(flux)

    with model:
        gp = model["obs"].owner.inputs[0]
        mu, var = gp.predict(trace.posterior.stack(draws=("chain", "draw")).mean(dim="draws").to_array(), return_var=True)
        std = np.sqrt(var)

    plt.figure(figsize=(10, 5))
    plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.5, label="Data")
    plt.plot(t, mu, color="C1", label="GP Mean")
    plt.fill_between(t, mu - std, mu + std, color="C1", alpha=0.3, label="1σ GP Uncertainty")
    plt.xlabel("Time [days]")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.title("Flexible GP Model Fit")
    plt.show()

    az.plot_trace(trace, var_names=["log_period", "log_amp", "log_sigma_trend", "log_rho_trend"])
    plt.show()
