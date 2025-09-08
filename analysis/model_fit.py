import numpy as np
import pandas as pd
import json
from scipy import optimize
from typing import Dict, Tuple, Optional


# =========================
# Utils
# =========================
# eps to avoid log(0) errors
_EPS = 1e-9


def best_minimize(loss, bounds, n_restarts=10, seed=42):
    """
    Find the best minimum of a loss function using random restarts (uniform over the bounds).
    """
    rng = np.random.default_rng(seed)
    best = None
    best_val = np.inf
    for _ in range(n_restarts):
        start = [rng.uniform(lo, hi) for lo, hi in bounds]
        res = optimize.minimize(loss, start, method="L-BFGS-B", bounds=bounds)
        if res.success and res.fun < best_val and not np.isnan(res.fun):
            best = res
            best_val = res.fun
    return best


def gaussian_nll(y_obs: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    """Gaussian negative log-likelihood."""
    if sigma <= 0:
        return np.inf
    r = y_obs - mu
    return 0.5 * np.sum(np.log(2 * np.pi * sigma**2) + (r**2) / (sigma**2))


def transform(x: np.ndarray, model_type: str, forward: bool = True) -> np.ndarray:
    """
    Transform the input data into the log space if needed.
    """
    if model_type == "weber":
        return (
            np.log(x + _EPS) if forward else (np.exp(x) - _EPS)
        )  # adding _EPS to prevent log(0)
    return x


def response_fit_space_and_jacobian(
    response: np.ndarray, model_type: str
) -> Tuple[np.ndarray, float]:
    """
    Returns (y_obs, jacobian_for_data_space_ll).
    For Weber (log-space fit): y_obs = log(response), jac = sum(log(response)).
    For Basic (data-space fit): y_obs = response, jac = 0.0.
    """
    if model_type == "weber":
        y_obs = np.log(response + _EPS)
        jac = float(np.sum(np.log(response + _EPS)))
    else:
        y_obs = response
        jac = 0.0
    return y_obs, jac


def sigma_upper(y: np.ndarray) -> float:
    """A reasonable upper bound for sigma based on the data."""
    s = float(np.std(y))
    return max(1e-3, 10.0 * s)


# =========================
# Data prep
# =========================
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by ensuring required columns are present,
    Drop rows with missing input / response values
    Create a run_id column that indicates which run a particular trial belongs to. (in the base case we have 5 runs)
    """

    df = df.copy()
    required = {"input_value", "response", "range_category", "trial", "stimulus_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=["input_value", "response"]).copy()
    df["range_id"] = pd.Categorical(df["range_category"]).codes.astype(int)

    def _within_range(g):
        s = g["stimulus_id"].astype(int).to_numpy()
        boundary = (s == 0) | (np.r_[True, s[1:] < s[:-1]])
        run_id = np.cumsum(boundary) - 1
        out = g.copy()
        out["run_id"] = run_id
        return out

    df = df.groupby("range_id", group_keys=False).apply(_within_range)
    return df


# =========================
# Models
# =========================
def fit_linear(
    input_stimulus: np.ndarray,
    response: np.ndarray,
    model_type: str = "basic",
    n_restarts: int = 10,  # restarts for fitting stability
) -> Dict:

    # transform data into fit space (log or not)
    x = transform(input_stimulus, model_type, forward=True)
    y_obs, jac = response_fit_space_and_jacobian(response, model_type)

    # OLS - initial parameter guess in fit space
    slope0, intercept0 = np.polyfit(x, y_obs, 1)
    sig0 = np.std(y_obs - (slope0 * x + intercept0)) + 1e-3

    # computes the log-likelihood
    def loss(theta):
        slope, intercept, sigma = theta
        mu = slope * x + intercept
        return gaussian_nll(y_obs, mu, sigma)

    bounds = [(-1e6, 1e6), (-1e6, 1e6), (1e-6, max(10 * sig0, 1e-3))]
    best = best_minimize(loss, bounds, n_restarts=n_restarts)

    # fall back to start optimization at OLS fitted parameters
    if best is None:
        res = optimize.minimize(
            loss, [slope0, intercept0, sig0], method="L-BFGS-B", bounds=bounds
        )
        if not res.success:
            raise RuntimeError(f"Linear_MLE({model_type}) failed")
        best = res

    slope, intercept, sigma = best.x
    mu_fit = slope * x + intercept
    nll_fit = float(loss([slope, intercept, sigma]))
    # AIC must use data-space likelihood (Weber gets Jacobian)
    nll_data = nll_fit + jac
    aic = 2 * len(bounds) + 2 * nll_data  # AIC is defined as 2*k + 2*nll

    # Predictions and reporting in data space
    pred = transform(mu_fit, model_type, forward=False)
    mse = float(np.mean((response - pred) ** 2))
    r2 = 1 - mse / np.var(response)

    return {
        "model": f"Linear_{model_type}",
        "params": {"slope": slope, "intercept": intercept, "sigma": sigma},
        "pred": pred,
        "aic": aic,
        "nll": nll_fit,  # fit-space NLL (diagnostic)
        "mse": mse,
        "r2": r2,
        "k": 3,
    }


def fit_static_bayes(
    input_stimulus: np.ndarray,
    response: np.ndarray,
    range_ids: np.ndarray,
    model_type: str = "basic",
    use_gain: bool = False,
    n_restarts: int = 10,
) -> Dict:
    """
    Fit a static Bayesian model to the data. In this model the prior is the mean of the input stimulus per range
    """

    x = transform(input_stimulus, model_type, forward=True)
    y_obs, jac = response_fit_space_and_jacobian(response, model_type)

    abs_max = float(np.max(np.abs(x)) + _EPS)

    # Core predictor in fit space (returns yhat_log for both; we'll inverse-transform later as needed)
    def predict_core(params):
        if use_gain:
            w_p, gain_factor, delta = params
        else:
            w_p, delta = params

        # mu_p mean input stimulus for each range
        mu_p = np.array([np.mean(x[range_ids == rid]) for rid in range_ids])
        yhat_log = (1 - w_p) * x + w_p * mu_p
        return (gain_factor * yhat_log + delta) if use_gain else (yhat_log + delta)

    def loss(theta):
        # theta ends with sigma, others are model params
        core, sigma = theta[:-1], theta[-1]
        if model_type == "weber":
            mu_fit = predict_core(core)  # already in log space
            return gaussian_nll(y_obs, mu_fit, sigma)  # fit-space NLL
        else:
            mu_fit = predict_core(
                core
            )  # "log" name reused; actually linear pre-inverse form
            mu_data = transform(mu_fit, model_type, forward=False)
            return gaussian_nll(y_obs, mu_data, sigma)  # here y_obs==response

    # Bounds and names
    if model_type == "weber":
        # [w_p, [gain_factor], delta, sigma]
        bounds = (
            [(0.0, 1.0), (0.1, 5.0), (-abs_max, abs_max), (1e-6, 10.0)]
            if use_gain
            else [(0.0, 1.0), (-abs_max, abs_max), (1e-6, 10.0)]
        )
    else:
        bounds = (
            [(0.0, 1.0), (0.1, 5.0), (-abs_max, abs_max), (1e-6, sigma_upper(response))]
            if use_gain
            else [(0.0, 1.0), (-abs_max, abs_max), (1e-6, sigma_upper(response))]
        )
    k = len(bounds)  # number of parameters
    names = (
        ["w_p", "gain_factor", "delta", "sigma"]
        if use_gain
        else ["w_p", "delta", "sigma"]
    )

    best = best_minimize(loss, bounds, n_restarts=n_restarts)
    if best is None:
        raise RuntimeError(f"Static_Bayes_MLE({model_type}) failed")
    theta = best.x
    nll_fit = float(loss(theta))

    # AIC with data-space likelihood
    nll_data = nll_fit + jac
    aic = 2 * k + 2 * nll_data

    core = theta[:-1]
    mu_fit = predict_core(core)
    pred = transform(mu_fit, model_type, forward=False)
    mse = float(np.mean((response - pred) ** 2))
    r2 = 1 - mse / np.var(response)

    return {
        "model": f"{'StaticGain' if use_gain else 'Static'}_{model_type}",
        "params": dict(zip(names, theta)),
        "pred": pred,
        "aic": aic,
        "nll": nll_fit,  # fit-space NLL
        "mse": mse,
        "r2": r2,
        "k": k,
    }


def predict_seq_bayes(
    input_stimulus, run_id, params, model_type="basic", use_gain=False
):
    """
    Predict the response using Kalman filter.
    """

    x = transform(input_stimulus, model_type, forward=True)
    if use_gain:
        r_div_q, gain_factor, delta = params
    else:
        r_div_q, delta = params

    preds = np.empty_like(x, dtype=float)

    # initialize mu and variance
    mu_t = var_t = prev_run = None
    for i, (xi, r) in enumerate(zip(x, run_id)):
        # reset if we are in the new run
        if r != prev_run:
            mu_t = xi
            var_t = 1e6  # vague prior
        var_pred = var_t + 1.0
        K_t = var_pred / (var_pred + r_div_q)
        mu_t = mu_t + K_t * (xi - mu_t)
        var_t = (1.0 - K_t) * var_pred
        yhat_log = (gain_factor * mu_t + delta) if use_gain else (mu_t + delta)
        preds[i] = transform(yhat_log, model_type, forward=False)
        prev_run = r
    return preds


def fit_seq_bayes(
    input_stimulus: np.ndarray,
    response: np.ndarray,
    run_id: np.ndarray,
    model_type: str = "basic",
    use_gain: bool = False,
    n_restarts: int = 10,
) -> Dict:
    y_obs, jac = response_fit_space_and_jacobian(response, model_type)
    abs_max = float(
        np.max(np.abs(transform(input_stimulus, model_type, forward=True))) + 1e-9
    )

    if use_gain:
        bounds = [
            (0.01, 100.0),
            (0.1, 5.0),
            (-abs_max, abs_max),
            (1e-6, sigma_upper(y_obs)),
        ]
        names = [
            "r_div_q",
            "gain_factor",
            "delta",
            "sigma_dec",
        ]  # r is the measurement noise and q is the process noise
        k = len(bounds)
    else:
        bounds = [(0.01, 100.0), (-abs_max, abs_max), (1e-6, sigma_upper(y_obs))]
        names = ["r_div_q", "delta", "sigma_dec"]
        k = len(bounds)

    def loss(theta):
        core, sigma = theta[:-1], theta[-1]
        pred = predict_seq_bayes(input_stimulus, run_id, core, model_type, use_gain)
        # Residuals in fit space
        if model_type == "weber":
            mu = np.log(pred + _EPS)  # pred is data-space; take log to fit-space
        else:
            mu = pred  # fit space == data space
        return gaussian_nll(y_obs, mu, sigma)

    best = best_minimize(loss, bounds, n_restarts=n_restarts)
    if best is None:
        raise RuntimeError("Seq_Bayes_MLE failed")
    theta = best.x
    nll_fit = float(loss(theta))
    nll_data = nll_fit + jac
    aic = 2 * k + 2 * nll_data

    pred = predict_seq_bayes(input_stimulus, run_id, theta[:-1], model_type, use_gain)
    mse = float(np.mean((response - pred) ** 2))
    r2 = 1 - mse / np.var(response)

    return {
        "model": f"{'SeqGain' if use_gain else 'Seq'}_{model_type}",
        "params": dict(zip(names, theta)),
        "pred": pred,
        "aic": aic,
        "nll": nll_fit,  # fit-space NLL
        "mse": mse,
        "r2": r2,
        "k": k,
    }


# =========================
# Generate summary file
# =========================


def w_p_conversion(r_div_q):
    """
    Derive the w_p (weight) from sequential model, which has the parameter r_div_q.
    """
    return r_div_q / (r_div_q + 1.0)


def summarize_results_df(
    results: dict, filepath: Optional[str] = None, top_n_per_scope: Optional[int] = None
) -> pd.DataFrame:
    """
    Flatten compare_models(...) output into a DataFrame with:
      scope, model, aic, weight, r2, mse, k, gain, w_p

    - gain: params['gain_factor'] when present, else NaN
    - w_p: if params['w_p'] exists, use it;
           elif params['r_div_q'] exists, compute r_div_q / (r_div_q + 1)
    - Computes ΔAIC and AIC weights within each scope
    - If top_n_per_scope is given, keeps only top N models per scope by weight
    - If filepath is given, saves DataFrame as CSV
    """
    rows = []

    def add_block(scope_label, model_dict):
        for mname, r in model_dict.items():
            p = r.get("params", {})
            rows.append(
                {
                    "scope": scope_label,
                    "model": r.get("model", mname),
                    "aic": float(r["aic"]),
                    "r2": float(r.get("r2", np.nan)),
                    "mse": float(r.get("mse", np.nan)),
                    "k": int(r.get("k", np.nan)),
                    "gain": float(p.get("gain_factor", np.nan)),
                    "w_p_raw": p.get("w_p", np.nan),
                    "r_div_q": p.get("r_div_q", np.nan),
                }
            )

    # Overall
    add_block("overall", results.get("overall", {}))
    # Per-range
    for rng, mdict in results.get("per_range", {}).items():
        add_block(f"range:{rng}", mdict)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def _derive_wp(row):
        """
        Derive the w_p (weight) from sequential model, which has the parameter r_div_q.
        """
        if pd.notna(row["w_p_raw"]):
            return float(row["w_p_raw"])
        if pd.notna(row["r_div_q"]):
            rdq = float(row["r_div_q"])
            return w_p_conversion(rdq)
        return np.nan

    df["w_p"] = df.apply(_derive_wp, axis=1)
    df.drop(columns=["w_p_raw", "r_div_q"], inplace=True)

    # Compute ΔAIC & weight
    df["delta_aic"] = df["aic"] - df.groupby("scope")["aic"].transform("min")
    exp_neg_half = np.exp(-0.5 * df["delta_aic"])
    denom = df.groupby("scope")["delta_aic"].transform(lambda s: np.exp(-0.5 * s).sum())
    df["weight"] = exp_neg_half / denom

    # Select columns
    df = df[
        [
            "scope",
            "model",
            "aic",
            "weight",
            "r2",
            "mse",
            "k",
            "gain",
            "w_p",
            "delta_aic",
        ]
    ]
    df = df.sort_values(["scope", "weight"], ascending=[True, False])

    # Keep top N if requested
    if top_n_per_scope is not None:
        df = df.groupby("scope", group_keys=False).head(int(top_n_per_scope))

    # Save if requested
    if filepath:
        df.to_csv(filepath, index=False)

    return df


# =========================
# Driver
# =========================
def compare_models(
    df: pd.DataFrame,
    model_types=("basic", "weber"),
    include_per_range: bool = True,
    save_results: bool = False,
    results_filename: str = "model_results.json",
) -> Dict[str, Dict]:

    df = preprocess_dataframe(
        df
    )  # clean up dataframe (remove na / missing and add a run_id

    results: Dict[str, Dict] = {"per_range": {}, "overall": {}}

    def fit_block(block_df: pd.DataFrame) -> Dict[str, Dict]:
        input_stimulus = block_df["input_value"].to_numpy()
        response = block_df["response"].to_numpy()
        range_ids = block_df["range_id"].to_numpy()
        run_id = block_df["run_id"].to_numpy()

        out: Dict[str, Dict] = {}
        for mt in model_types:
            out[f"Linear_{mt}"] = fit_linear(input_stimulus, response, mt)
            out[f"Static_{mt}"] = fit_static_bayes(
                input_stimulus, response, range_ids, mt, use_gain=False
            )
            out[f"StaticGain_{mt}"] = fit_static_bayes(
                input_stimulus, response, range_ids, mt, use_gain=True
            )
            out[f"Seq_{mt}"] = fit_seq_bayes(
                input_stimulus, response, run_id, mt, use_gain=False
            )
            out[f"SeqGain_{mt}"] = fit_seq_bayes(
                input_stimulus, response, run_id, mt, use_gain=True
            )
        return out

    if include_per_range:
        for rng, g in df.groupby("range_category", sort=False):
            results["per_range"][rng] = fit_block(g)

    results["overall"] = fit_block(df)

    if save_results:
        with open(results_filename, "w") as f:
            json.dump(results, f, indent=4, default=str)

    return results
