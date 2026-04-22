"""Core Panel-GARCH-X model with custom MLE.

Implements a GARCH(1,1)-X model estimated jointly across a panel of stocks
via maximum likelihood. Parameters can be individually configured as either
global (shared across all stocks) or stock-specific.

Model for stock i at time t:
    Mean:     r_{i,t} = ε_{i,t}               (zero-mean)
    Variance: σ²_{i,t} = ω_i + α · ε²_{i,t-1} + β · σ²_{i,t-1} + δ' · x_{i,t-1}
    ε_{i,t} | F_{t-1}  ~ D(0, σ²_{i,t})

where D is Normal or Student-t.
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from config import GARCHSpec
from numba import njit
from scipy import optimize
from scipy.special import gammaln

from data import StockData

# =============================================================================
# Parameter Layout
# =============================================================================


@dataclass
class ParamLayout:
    """Maps parameter names to positions in the flat optimization vector.

    Handles the mixed global/stock-specific structure:
    - Global params occupy one slot each
    - Stock-specific params occupy N slots (one per stock)
    """

    n_stocks: int
    n_exog: int
    spec: GARCHSpec
    exog_names: List[str] = field(default_factory=list)

    # Computed indices (populated by __post_init__)
    slices: Dict[str, Optional[slice]] = field(default_factory=dict, init=False)
    n_params: int = field(default=0, init=False)
    param_names: List[str] = field(default_factory=list, init=False)
    # Per-column delta layout: one entry per exog column. Each slice covers
    # either 1 slot (pooled) or N_stocks slots (stock-specific).
    delta_col_slices: List[slice] = field(default_factory=list, init=False)
    delta_col_pooled: List[bool] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        idx = 0

        # mu
        if self.spec.mu_scope == "zero":
            self.slices["mu"] = None  # no parameter
        elif self.spec.mu_scope == "global":
            self.slices["mu"] = slice(idx, idx + 1)
            self.param_names.append("mu")
            idx += 1
        else:  # stock
            self.slices["mu"] = slice(idx, idx + self.n_stocks)
            self.param_names.extend([f"mu_{i}" for i in range(self.n_stocks)])
            idx += self.n_stocks

        # omega
        n = 1 if self.spec.omega_scope == "global" else self.n_stocks
        self.slices["omega"] = slice(idx, idx + n)
        if n == 1:
            self.param_names.append("omega")
        else:
            self.param_names.extend([f"omega_{i}" for i in range(self.n_stocks)])
        idx += n

        # alpha
        n = 1 if self.spec.alpha_scope == "global" else self.n_stocks
        self.slices["alpha"] = slice(idx, idx + n)
        if n == 1:
            self.param_names.append("alpha")
        else:
            self.param_names.extend([f"alpha_{i}" for i in range(self.n_stocks)])
        idx += n

        # beta
        n = 1 if self.spec.beta_scope == "global" else self.n_stocks
        self.slices["beta"] = slice(idx, idx + n)
        if n == 1:
            self.param_names.append("beta")
        else:
            self.param_names.extend([f"beta_{i}" for i in range(self.n_stocks)])
        idx += n

        # delta (one per exogenous variable; each column may be globally
        # pooled or stock-specific independently).
        if self.n_exog > 0:
            col_labels = (
                self.exog_names
                if self.exog_names
                else [str(k) for k in range(self.n_exog)]
            )
            delta_start = idx
            for col in col_labels:
                is_pooled = self.spec.is_delta_col_pooled(col)
                n_per = 1 if is_pooled else self.n_stocks
                col_slice = slice(idx, idx + n_per)
                self.delta_col_slices.append(col_slice)
                self.delta_col_pooled.append(is_pooled)
                if is_pooled:
                    self.param_names.append(f"delta_{col}")
                else:
                    self.param_names.extend(
                        [f"delta_{col}_stock{i}" for i in range(self.n_stocks)]
                    )
                idx += n_per
            self.slices["delta"] = slice(delta_start, idx)
        else:
            self.slices["delta"] = None

        # nu (degrees of freedom for Student-t)
        if self.spec.dist == "studentt":
            self.slices["nu"] = slice(idx, idx + 1)
            self.param_names.append("nu")
            idx += 1

        self.n_params = idx

    def get_mu(self, theta: np.ndarray, stock_idx: int) -> float:
        if self.spec.mu_scope == "zero":
            return 0.0
        s = self.slices["mu"]
        arr = theta[s]
        return arr[0] if self.spec.mu_scope == "global" else arr[stock_idx]

    def get_omega(self, theta: np.ndarray, stock_idx: int) -> float:
        s = self.slices["omega"]
        arr = theta[s]
        return arr[0] if self.spec.omega_scope == "global" else arr[stock_idx]

    def get_alpha(self, theta: np.ndarray, stock_idx: int) -> float:
        s = self.slices["alpha"]
        arr = theta[s]
        return arr[0] if self.spec.alpha_scope == "global" else arr[stock_idx]

    def get_beta(self, theta: np.ndarray, stock_idx: int) -> float:
        s = self.slices["beta"]
        arr = theta[s]
        return arr[0] if self.spec.beta_scope == "global" else arr[stock_idx]

    def get_delta(self, theta: np.ndarray, stock_idx: int) -> np.ndarray:
        """Return (K,) vector of exogenous coefficients for a stock."""
        if self.slices["delta"] is None:
            return np.array([])
        out = np.empty(self.n_exog)
        for k, (col_slice, is_pooled) in enumerate(
            zip(self.delta_col_slices, self.delta_col_pooled)
        ):
            if is_pooled:
                out[k] = theta[col_slice.start]
            else:
                out[k] = theta[col_slice.start + stock_idx]
        return out

    def get_nu(self, theta: np.ndarray) -> float:
        if self.spec.dist != "studentt":
            return np.inf
        return theta[self.slices["nu"]][0]


# =============================================================================
# Likelihood Computation
# =============================================================================


@njit(cache=True)
def _garch_filter_core(
    eps: np.ndarray,
    exog: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    delta: np.ndarray,
    init_var: float,
) -> np.ndarray:
    """Inner GARCH(1,1)-X variance recursion (numba-accelerated)."""
    T = len(eps)
    sigma2 = np.empty(T)
    sigma2[0] = init_var
    K = len(delta)
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        if K > 0:
            for k in range(K):
                sigma2[t] += delta[k] * exog[t - 1, k]
        if sigma2[t] < 1e-12:
            sigma2[t] = 1e-12
    return sigma2


def _garch_filter_stock(
    returns: np.ndarray,
    exog: np.ndarray,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    delta: np.ndarray,
    init_var: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the GARCH(1,1)-X variance filter for a single stock.

    Returns
    -------
    sigma2 : (T,) conditional variances
    eps : (T,) demeaned returns (innovations)
    """
    eps = returns - mu
    sigma2 = _garch_filter_core(eps, exog, omega, alpha, beta, delta, init_var)
    return sigma2, eps


def _normal_loglik(eps: np.ndarray, sigma2: np.ndarray) -> float:
    """Gaussian log-likelihood (excluding constant)."""
    return -0.5 * np.sum(np.log(sigma2) + eps**2 / sigma2)


def _studentt_loglik(eps: np.ndarray, sigma2: np.ndarray, nu: float) -> float:
    """Student-t log-likelihood."""
    if nu <= 2.0:
        return -1e20  # invalid
    T = len(eps)
    half_nu = 0.5 * nu
    half_nup1 = 0.5 * (nu + 1)
    const = gammaln(half_nup1) - gammaln(half_nu) - 0.5 * np.log(np.pi * (nu - 2))
    ll = T * const - 0.5 * np.sum(
        np.log(sigma2) + (nu + 1) * np.log(1 + eps**2 / (sigma2 * (nu - 2)))
    )
    return ll


def panel_loglikelihood(
    theta: np.ndarray,
    stocks: List[StockData],
    layout: ParamLayout,
    spec: GARCHSpec,
) -> float:
    """Compute the total panel log-likelihood (to be MAXIMIZED).

    Returns negative infinity if parameters are infeasible.
    """
    total_ll = 0.0
    nu = layout.get_nu(theta) if spec.dist == "studentt" else np.inf

    for i, stock in enumerate(stocks):
        mu = layout.get_mu(theta, i)
        omega = layout.get_omega(theta, i)
        alpha = layout.get_alpha(theta, i)
        beta_ = layout.get_beta(theta, i)
        delta = layout.get_delta(theta, i)

        # Feasibility checks
        if omega <= 0 or alpha < 0 or beta_ < 0 or alpha + beta_ >= 1.0:
            return -np.inf

        init_var = (
            stock.sample_var if spec.var_init == "unconditional" else stock.sample_var
        )
        sigma2, eps = _garch_filter_stock(
            stock.returns, stock.exog, mu, omega, alpha, beta_, delta, init_var
        )

        # Skip first observation (used for initialization)
        sigma2 = sigma2[1:]
        eps = eps[1:]

        if spec.dist == "normal":
            total_ll += _normal_loglik(eps, sigma2)
        else:
            total_ll += _studentt_loglik(eps, sigma2, nu)

    return total_ll


class _ProgressTracker:
    """Tracks and prints progress during optimization."""

    def __init__(self, verbose: bool = True, print_every: int = 10):
        self.verbose = verbose
        self.print_every = print_every
        self.n_eval = 0
        self.best_ll = -np.inf
        self.t_start = time.time()

    def reset(self) -> None:
        self.n_eval = 0
        self.best_ll = -np.inf
        self.t_start = time.time()

    def report(self, ll: float) -> None:
        self.n_eval += 1
        if ll > self.best_ll:
            self.best_ll = ll
        if self.verbose and self.n_eval % self.print_every == 0:
            elapsed = time.time() - self.t_start
            sys.stdout.write(
                f"\r  eval {self.n_eval:>5d} | "
                f"best LL = {self.best_ll:>14,.2f} | "
                f"{elapsed:>6.1f}s"
            )
            sys.stdout.flush()

    def finish(self) -> None:
        if self.verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()


# Global tracker instance (set before each optimization run)
_tracker = _ProgressTracker(verbose=False)


def _neg_loglik(theta: np.ndarray, stocks, layout, spec) -> float:
    """Negative log-likelihood for minimization."""
    ll = panel_loglikelihood(theta, stocks, layout, spec)
    if not np.isfinite(ll):
        _tracker.report(-1e20)
        return 1e20
    _tracker.report(ll)
    return -ll


# =============================================================================
# Initial Values
# =============================================================================


def compute_initial_values(
    stocks: List[StockData],
    layout: ParamLayout,
    spec: GARCHSpec,
) -> np.ndarray:
    """Compute reasonable starting values for optimization."""
    theta0 = np.zeros(layout.n_params)

    # Typical GARCH(1,1) starting values
    default_alpha = 0.05
    default_beta = 0.90
    avg_var = np.mean([s.sample_var for s in stocks])

    for i in range(len(stocks)):
        # mu
        if spec.mu_scope != "zero":
            s = layout.slices["mu"]
            assert s is not None
            if spec.mu_scope == "global":
                theta0[s] = np.mean([np.mean(s_.returns) for s_ in stocks])
            else:
                theta0[s.start + i] = np.mean(stocks[i].returns)

        # omega  ≈  var * (1 - alpha - beta)
        omega_init = stocks[i].sample_var * (1 - default_alpha - default_beta)
        s = layout.slices["omega"]
        assert s is not None
        if spec.omega_scope == "global":
            theta0[s] = avg_var * (1 - default_alpha - default_beta)
        else:
            theta0[s.start + i] = omega_init

        # alpha
        s = layout.slices["alpha"]
        assert s is not None
        if spec.alpha_scope == "global":
            theta0[s] = default_alpha
        else:
            theta0[s.start + i] = default_alpha

        # beta
        s = layout.slices["beta"]
        assert s is not None
        if spec.beta_scope == "global":
            theta0[s] = default_beta
        else:
            theta0[s.start + i] = default_beta

    # delta: start at zero (no exogenous effect)
    if layout.slices["delta"] is not None:
        theta0[layout.slices["delta"]] = 0.0

    # nu for Student-t (start at 8)
    if spec.dist == "studentt":
        theta0[layout.slices["nu"]] = 8.0

    return theta0


# =============================================================================
# Bounds
# =============================================================================


def compute_bounds(
    layout: ParamLayout,
    spec: GARCHSpec,
) -> List[Tuple[Optional[float], Optional[float]]]:
    """Compute parameter bounds for L-BFGS-B / SLSQP."""
    bounds: List[Tuple[Optional[float], Optional[float]]] = [
        (None, None)
    ] * layout.n_params

    # mu: unconstrained
    if spec.mu_scope != "zero" and layout.slices["mu"] is not None:
        s = layout.slices["mu"]
        assert s is not None
        for j in range(s.start, s.stop):
            bounds[j] = (-0.1, 0.1)

    # omega > 0
    s = layout.slices["omega"]
    assert s is not None
    for j in range(s.start, s.stop):
        bounds[j] = (1e-10, None)

    # alpha >= 0
    s = layout.slices["alpha"]
    assert s is not None
    for j in range(s.start, s.stop):
        bounds[j] = (1e-6, 0.999)

    # beta >= 0
    s = layout.slices["beta"]
    assert s is not None
    for j in range(s.start, s.stop):
        bounds[j] = (1e-6, 0.999)

    # delta: unconstrained (social media can increase or decrease volatility)
    if layout.slices["delta"] is not None:
        s = layout.slices["delta"]
        for j in range(s.start, s.stop):
            bounds[j] = (None, None)

    # nu > 2 for Student-t
    if spec.dist == "studentt":
        s = layout.slices["nu"]
        assert s is not None
        bounds[s.start] = (2.1, 100.0)

    return bounds


# =============================================================================
# Estimation
# =============================================================================


@dataclass
class PanelGARCHResult:
    """Results from Panel-GARCH-X estimation."""

    spec: GARCHSpec
    layout: ParamLayout
    theta: np.ndarray  # optimized parameter vector
    loglik: float  # maximized log-likelihood
    n_obs: int  # total observations (all stocks)
    n_stocks: int
    n_params: int
    aic: float
    bic: float
    converged: bool
    hessian_inv: Optional[np.ndarray]  # inverse Hessian (for std errors)
    std_errors: Optional[np.ndarray]  # parameter standard errors
    elapsed_sec: float
    stock_tickers: List[str]

    def summary_dict(self) -> Dict[str, float]:
        """Return {param_name: estimate} dictionary."""
        return dict(zip(self.layout.param_names, self.theta))

    def get_conditional_variances(
        self, stocks: List[StockData]
    ) -> Dict[str, np.ndarray]:
        """Compute fitted conditional variances for each stock."""
        result = {}
        for i, stock in enumerate(stocks):
            mu = self.layout.get_mu(self.theta, i)
            omega = self.layout.get_omega(self.theta, i)
            alpha = self.layout.get_alpha(self.theta, i)
            beta_ = self.layout.get_beta(self.theta, i)
            delta = self.layout.get_delta(self.theta, i)
            init_var = stock.sample_var
            sigma2, _ = _garch_filter_stock(
                stock.returns, stock.exog, mu, omega, alpha, beta_, delta, init_var
            )
            result[stock.ticker] = sigma2
        return result


def estimate_panel_garch(
    stocks: List[StockData],
    spec: GARCHSpec,
    theta0: Optional[np.ndarray] = None,
    skip_se: bool = False,
) -> PanelGARCHResult:
    """Estimate the Panel-GARCH-X model via maximum likelihood.

    Uses block coordinate descent when there are both global and stock-specific
    parameters (the typical case). This alternates:
      1. Optimize global params (alpha, beta, delta) given fixed stock omegas — low-dim
      2. Optimize each stock's omega given fixed global params — N independent 1D problems

    Falls back to joint optimization for small problems or all-global specs.

    Parameters
    ----------
    stocks : list of StockData
        Prepared per-stock data.
    spec : GARCHSpec
        Model specification.
    theta0 : array, optional
        Starting values. If None, computed automatically.
    skip_se : bool
        If True, skip standard error computation (much faster).

    Returns
    -------
    PanelGARCHResult
    """
    N = len(stocks)
    K = len(spec.exog_cols)
    layout = ParamLayout(
        n_stocks=N, n_exog=K, spec=spec, exog_names=list(spec.exog_cols)
    )

    has_stock_delta = any(not p for p in layout.delta_col_pooled)
    has_stock_params = (
        spec.omega_scope == "stock"
        or spec.alpha_scope == "stock"
        or spec.beta_scope == "stock"
        or has_stock_delta
        or (spec.mu_scope == "stock")
    )
    # Use BCD for large problems with mixed global/stock-specific params
    use_bcd = has_stock_params and N > 20

    if spec.verbose:
        print(f"\n{'='*60}")
        print(spec.describe())
        print(f"  Stocks:      {N}")
        print(f"  Total obs:   {sum(s.n_obs for s in stocks):,}")
        print(f"  Parameters:  {layout.n_params}")
        print(
            f"  Optimizer:   {'block coordinate descent' if use_bcd else 'joint L-BFGS-B'}"
        )
        print(f"{'='*60}\n")

    # Warm up numba JIT on first call (compile once)
    if len(stocks) > 0:
        _s = stocks[0]
        _ = _garch_filter_core(
            _s.returns[:10] - np.mean(_s.returns[:10]),
            _s.exog[:10],
            _s.sample_var * 0.05,
            0.05,
            0.90,
            np.zeros(K),
            _s.sample_var,
        )

    if theta0 is None:
        theta0 = compute_initial_values(stocks, layout, spec)

    # Check initial log-likelihood
    init_ll = panel_loglikelihood(theta0, stocks, layout, spec)
    if spec.verbose:
        print(f"  Initial log-likelihood: {init_ll:,.2f}", flush=True)

    t_start = time.time()

    if use_bcd:
        theta_hat, final_ll, converged = _bcd_optimize(
            theta0, stocks, layout, spec, t_start
        )
    else:
        theta_hat, final_ll, converged = _joint_optimize(
            theta0, stocks, layout, spec, t_start
        )

    elapsed = time.time() - t_start
    loglik = final_ll
    n_obs_total = sum(s.n_obs - 1 for s in stocks)

    # Information criteria
    k = layout.n_params
    aic = -2 * loglik + 2 * k
    bic = -2 * loglik + k * np.log(n_obs_total)

    # Standard errors (only for global params in BCD mode to keep it fast)
    std_errors = None
    hessian_inv = None
    if not skip_se:
        std_errors = _compute_standard_errors(
            theta_hat,
            stocks,
            layout,
            spec,
            use_bcd,
            verbose=spec.verbose,
        )

    if spec.verbose:
        print(f"\n  Final log-likelihood:   {loglik:,.2f}")
        print(f"  AIC: {aic:,.2f}  |  BIC: {bic:,.2f}")
        print(f"  Converged: {converged}")
        print(f"  Elapsed: {elapsed:.1f}s")

    return PanelGARCHResult(
        spec=spec,
        layout=layout,
        theta=theta_hat,
        loglik=loglik,
        n_obs=n_obs_total,
        n_stocks=N,
        n_params=k,
        aic=aic,
        bic=bic,
        converged=converged,
        hessian_inv=hessian_inv,
        std_errors=std_errors,
        elapsed_sec=elapsed,
        stock_tickers=[s.ticker for s in stocks],
    )


# =============================================================================
# Joint optimization (small problems)
# =============================================================================


def _joint_optimize(
    theta0: np.ndarray,
    stocks: List[StockData],
    layout: ParamLayout,
    spec: GARCHSpec,
    t_start: float,
) -> Tuple[np.ndarray, float, bool]:
    """Standard joint optimization via L-BFGS-B."""
    bounds = compute_bounds(layout, spec)

    global _tracker
    _tracker = _ProgressTracker(verbose=spec.verbose, print_every=5)
    _tracker.t_start = t_start

    result = optimize.minimize(
        _neg_loglik,
        theta0,
        args=(stocks, layout, spec),
        method=spec.optimizer,
        bounds=bounds,
        options={"maxiter": spec.maxiter, "ftol": spec.tol, "disp": False},
    )
    _tracker.finish()

    return result.x, -result.fun, result.success


# =============================================================================
# Block Coordinate Descent (large panels with mixed global/stock params)
# =============================================================================


def _stock_loglik(
    stock: StockData,
    mu: float,
    omega: float,
    alpha: float,
    beta_: float,
    delta: np.ndarray,
    dist: str,
    nu: float = np.inf,
) -> float:
    """Log-likelihood for a single stock given all parameters."""
    sigma2, eps = _garch_filter_stock(
        stock.returns, stock.exog, mu, omega, alpha, beta_, delta, stock.sample_var
    )
    sigma2 = sigma2[1:]
    eps = eps[1:]
    if dist == "normal":
        return _normal_loglik(eps, sigma2)
    else:
        return _studentt_loglik(eps, sigma2, nu)


def _bcd_optimize(
    theta0: np.ndarray,
    stocks: List[StockData],
    layout: ParamLayout,
    spec: GARCHSpec,
    t_start: float,
    max_rounds: int = 60,
    tol: float = 1e-2,
) -> Tuple[np.ndarray, float, bool]:
    """Block coordinate descent: alternate global and stock-specific blocks.

    Block 1 (Global): Optimize (alpha, beta, delta, [nu]) given fixed stock params
    Block 2 (Stock):  For each stock, optimize (omega_i, [mu_i]) given fixed global params
    """
    theta = theta0.copy()
    N = len(stocks)
    K = layout.n_exog

    prev_ll = panel_loglikelihood(theta, stocks, layout, spec)
    # Always print BCD progress (even when verbose=False) — these runs can be long
    print(f"  BCD round   0 | LL = {prev_ll:>14,.2f}", flush=True)

    converged = False
    for rnd in range(1, max_rounds + 1):
        rnd_start = time.time()

        # --- Block 1: Optimize global params given fixed stock-specific params ---
        global_idx = _get_global_indices(layout, spec)
        if len(global_idx) > 0:
            theta = _optimize_global_block(theta, global_idx, stocks, layout, spec)

        # --- Block 2: Optimize stock-specific params given fixed global params ---
        theta = _optimize_stock_block(theta, stocks, layout, spec)

        # Check convergence
        current_ll = panel_loglikelihood(theta, stocks, layout, spec)
        improvement = current_ll - prev_ll
        rnd_elapsed = time.time() - rnd_start
        total_elapsed = time.time() - t_start

        print(
            f"  BCD round {rnd:>3d} | LL = {current_ll:>14,.2f} | "
            f"delta = {improvement:>+10,.4f} | "
            f"{rnd_elapsed:.1f}s (total {total_elapsed:.1f}s)",
            flush=True,
        )

        if abs(improvement) < tol and rnd > 1:
            converged = True
            print(f"  Converged after {rnd} rounds.", flush=True)
            break

        prev_ll = current_ll

    final_ll = panel_loglikelihood(theta, stocks, layout, spec)
    return theta, final_ll, converged


def _get_global_indices(layout: ParamLayout, spec: GARCHSpec) -> List[int]:
    """Get indices in theta that correspond to global parameters."""
    indices = []
    for param_name, scope in [
        ("alpha", spec.alpha_scope),
        ("beta", spec.beta_scope),
    ]:
        if scope == "global":
            s = layout.slices[param_name]
            if s is not None:
                indices.extend(range(s.start, s.stop))

    if spec.delta_scope == "global" or spec.delta_pooled_cols is not None:
        # Include only the delta columns that are pooled.
        for col_slice, is_pooled in zip(
            layout.delta_col_slices, layout.delta_col_pooled
        ):
            if is_pooled:
                indices.extend(range(col_slice.start, col_slice.stop))

    if spec.mu_scope == "global":
        s = layout.slices.get("mu")
        if s is not None:
            indices.extend(range(s.start, s.stop))

    if spec.omega_scope == "global":
        s = layout.slices["omega"]
        if s is not None:
            indices.extend(range(s.start, s.stop))

    if spec.dist == "studentt":
        s = layout.slices.get("nu")
        if s is not None:
            indices.extend(range(s.start, s.stop))

    return sorted(indices)


def _optimize_global_block(
    theta: np.ndarray,
    global_idx: List[int],
    stocks: List[StockData],
    layout: ParamLayout,
    spec: GARCHSpec,
) -> np.ndarray:
    """Optimize only the global parameters, keeping stock-specific ones fixed."""
    full_bounds = compute_bounds(layout, spec)
    sub_bounds = [full_bounds[i] for i in global_idx]
    x0 = theta[global_idx].copy()

    def _sub_neg_ll(x_global):
        theta_trial = theta.copy()
        theta_trial[global_idx] = x_global
        ll = panel_loglikelihood(theta_trial, stocks, layout, spec)
        return -ll if np.isfinite(ll) else 1e20

    result = optimize.minimize(
        _sub_neg_ll,
        x0,
        method="L-BFGS-B",
        bounds=sub_bounds,
        options={"maxiter": 200, "ftol": 1e-10},
    )

    theta_new = theta.copy()
    theta_new[global_idx] = result.x
    return theta_new


def _optimize_stock_block(
    theta: np.ndarray,
    stocks: List[StockData],
    layout: ParamLayout,
    spec: GARCHSpec,
) -> np.ndarray:
    """Optimize stock-specific params for each stock (independent 1D/low-dim problems)."""
    theta_new = theta.copy()
    full_bounds = compute_bounds(layout, spec)

    for i, stock in enumerate(stocks):
        # Collect indices of stock-specific params for stock i
        stock_idx = []
        if spec.omega_scope == "stock":
            s = layout.slices["omega"]
            if s is not None:
                stock_idx.append(s.start + i)
        if spec.mu_scope == "stock":
            s = layout.slices.get("mu")
            if s is not None:
                stock_idx.append(s.start + i)
        if spec.alpha_scope == "stock":
            s = layout.slices["alpha"]
            if s is not None:
                stock_idx.append(s.start + i)
        if spec.beta_scope == "stock":
            s = layout.slices["beta"]
            if s is not None:
                stock_idx.append(s.start + i)
        # Stock-specific delta entries: one per non-pooled column for stock i.
        for col_slice, is_pooled in zip(
            layout.delta_col_slices, layout.delta_col_pooled
        ):
            if not is_pooled:
                stock_idx.append(col_slice.start + i)

        if not stock_idx:
            continue

        sub_bounds = [full_bounds[j] for j in stock_idx]
        x0 = theta_new[stock_idx].copy()

        # Extract fixed global params for this stock
        def _make_stock_neg_ll(si, sidx):
            def _f(x_stock):
                theta_trial = theta_new.copy()
                theta_trial[sidx] = x_stock
                mu = layout.get_mu(theta_trial, si)
                omega = layout.get_omega(theta_trial, si)
                alpha = layout.get_alpha(theta_trial, si)
                beta_ = layout.get_beta(theta_trial, si)
                delta = layout.get_delta(theta_trial, si)
                nu = layout.get_nu(theta_trial) if spec.dist == "studentt" else np.inf
                if omega <= 0 or alpha < 0 or beta_ < 0 or alpha + beta_ >= 1.0:
                    return 1e20
                ll = _stock_loglik(stock, mu, omega, alpha, beta_, delta, spec.dist, nu)
                return -ll if np.isfinite(ll) else 1e20

            return _f

        obj = _make_stock_neg_ll(i, stock_idx)

        if len(stock_idx) == 1:
            # Use bounded scalar optimization (Brent) — very fast
            lo = sub_bounds[0][0] if sub_bounds[0][0] is not None else 1e-12
            hi = sub_bounds[0][1] if sub_bounds[0][1] is not None else x0[0] * 100
            hi = max(hi, lo + 1e-6)
            res = optimize.minimize_scalar(
                lambda x: obj(np.array([x])),
                bounds=(lo, hi),
                method="bounded",
                options={"maxiter": 100, "xatol": 1e-12},
            )
            theta_new[stock_idx[0]] = res.x  # type: ignore[union-attr]
        else:
            # maxiter chosen to allow inner solve to make meaningful
            # progress on all directions (notably beta) per BCD round.
            res = optimize.minimize(
                obj,
                x0,
                method="L-BFGS-B",
                bounds=sub_bounds,
                options={"maxiter": 100, "ftol": 1e-10},
            )
            theta_new[stock_idx] = res.x

    return theta_new


# =============================================================================
# Standard Errors
# =============================================================================


def _compute_standard_errors(
    theta: np.ndarray,
    stocks: List[StockData],
    layout: ParamLayout,
    spec: GARCHSpec,
    bcd_mode: bool,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """Compute standard errors.

    In BCD mode with many stock-specific params, compute SE only for global
    params via profile likelihood Hessian (much faster than full Hessian).
    For small problems, compute full Hessian.
    """
    global_idx = _get_global_indices(layout, spec)
    n_total = layout.n_params

    if bcd_mode and len(global_idx) < n_total:
        # Only compute SE for global parameters
        if verbose:
            ng = len(global_idx)
            print(
                f"  Computing std errors for {ng} global params "
                f"({ng*(ng+1)//2} Hessian evals)...",
                flush=True,
            )

        def _profile_neg_ll(x_global):
            theta_trial = theta.copy()
            theta_trial[global_idx] = x_global
            ll = panel_loglikelihood(theta_trial, stocks, layout, spec)
            return -ll if np.isfinite(ll) else 1e20

        try:
            x_global = theta[global_idx]
            hess = _numerical_hessian(
                _profile_neg_ll,
                x_global,
                verbose=verbose,
            )
            hess_inv = np.linalg.inv(hess)
            se_sq = np.diag(hess_inv)
            se_sq = np.clip(se_sq, 0, None)
            se_global = np.sqrt(se_sq)

            # Build full SE vector (NaN for stock-specific params)
            full_se = np.full(n_total, np.nan)
            for j, gidx in enumerate(global_idx):
                full_se[gidx] = se_global[j]
            return full_se
        except Exception as e:
            warnings.warn(f"Could not compute standard errors: {e}")
            return None
    else:
        if verbose:
            print(
                f"  Computing Hessian ({n_total}x{n_total} = "
                f"{n_total*(n_total+1)//2} evaluations)...",
                flush=True,
            )
        try:
            hess = _numerical_hessian(
                _neg_loglik,
                theta,
                args=(stocks, layout, spec),
                verbose=verbose,
            )
            hess_inv = np.linalg.inv(hess)
            se_sq = np.diag(hess_inv)
            se_sq = np.clip(se_sq, 0, None)
            return np.sqrt(se_sq)
        except Exception as e:
            warnings.warn(f"Could not compute standard errors: {e}")
            return None


def _numerical_hessian(
    func,
    x0: np.ndarray,
    args=(),
    eps: float = 1e-5,
    verbose: bool = False,
) -> np.ndarray:
    """Compute Hessian via central finite differences."""
    n = len(x0)
    total_rows = n
    H = np.zeros((n, n))
    f0 = func(x0, *args)
    ei = np.zeros(n)
    ej = np.zeros(n)

    t0 = time.time()
    for i in range(n):
        ei[i] = eps
        if verbose and (i + 1) % max(1, n // 20) == 0:
            elapsed = time.time() - t0
            sys.stdout.write(
                f"\r  Hessian row {i+1}/{n} ({100*(i+1)/n:.0f}%) | {elapsed:.1f}s"
            )
            sys.stdout.flush()
        for j in range(i, n):
            ej[j] = eps
            fpp = func(x0 + ei + ej, *args)
            fpm = func(x0 + ei - ej, *args)
            fmp = func(x0 - ei + ej, *args)
            fmm = func(x0 - ei - ej, *args)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps**2)
            H[j, i] = H[i, j]
            ej[j] = 0.0
        ei[i] = 0.0

    if verbose:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return H
