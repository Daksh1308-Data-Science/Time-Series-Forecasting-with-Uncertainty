"""Bayesian demand forecasting: posterior predictive intervals.

Model
-----
A Bayesian structural demand model on the weekly series:

    y_t  ~  Normal(mu_t, sigma)
    mu_t = const + beta_trend * t + sum_k [a_k cos(2 pi k t / 52)
                                           + b_k sin(2 pi k t / 52)]
                + sum_h gamma_h * 1[week t is holiday h]

The Fourier terms capture the 52-week retail cycle; the holiday indicators
capture the four Walmart markdown weeks, which are additive spikes that no
smooth seasonal term can represent.

Two engines, one interface
--------------------------
* ``pymc``      - NUTS sampling of the posterior (parameter uncertainty from
                  the actual posterior, hyperparameters learned by MCMC).
* ``closed_form`` - the exact conjugate posterior for this Normal linear model
                  with a reference prior p(beta, sigma^2) ~ 1 / sigma^2:
                  sigma^2 | y ~ InvGamma(nu/2, nu*s^2/2),
                  beta | sigma^2, y ~ N(beta_hat, sigma^2 (X'X)^-1).
                  No MCMC needed and it is still genuinely Bayesian.

``engine="auto"`` uses PyMC when importable and silently falls back to the
closed form otherwise, so the dashboard and tests never break on a machine
where PyMC cannot be installed. Both engines return posterior *predictive*
samples, so intervals include observation noise, not just parameter spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .holidays import holiday_indicators
from .schema import band_columns, future_dates, level_tag

DEFAULT_LEVELS = (0.80, 0.95)
ENGINES = ("auto", "pymc", "closed_form")


def resolve_engine(engine: str = "auto") -> str:
    """Map a requested engine to one that is actually available.

    ``"pymc"`` requested but not importable falls back to ``"closed_form"``
    instead of raising: an optional dependency must never break the app (the
    Cloud deployment has no PyMC by design). Callers can compare
    ``engine_requested`` with ``engine`` to tell the user what happened.
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    if engine == "closed_form":
        return "closed_form"
    try:
        import pymc  # noqa: F401

        return "pymc"
    except Exception:  # noqa: BLE001 - any import/ABI failure means "not available"
        return "closed_form"


def build_design(
    t: np.ndarray,
    denom: float,
    period: int = 52,
    order: int = 6,
    holidays: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Design matrix: intercept, scaled trend, Fourier pairs, holiday dummies."""
    t = np.asarray(t, dtype=float)
    columns: list[np.ndarray] = [np.ones(t.size), t / denom]
    names = ["const", "trend"]
    for k in range(1, order + 1):
        columns.append(np.cos(2 * np.pi * k * t / period))
        columns.append(np.sin(2 * np.pi * k * t / period))
        names += [f"cos{k}", f"sin{k}"]
    if holidays is not None and holidays.shape[1] > 0:
        for column in holidays.columns:
            columns.append(np.asarray(holidays[column], dtype=float))
            names.append(f"holiday_{column}")
    return np.column_stack(columns), names


class BayesianForecast:
    """Fit once, forecast many times (with posterior predictive bands)."""

    def __init__(
        self,
        df: pd.DataFrame,
        period: int = 52,
        order: int = 6,
        holiday_dates: dict[str, list[str]] | None = None,
        engine: str = "auto",
        seed: int = 42,
        draws: int = 1500,
        tune: int = 1500,
        chains: int = 2,
        levels=DEFAULT_LEVELS,
        log_target: bool = False,
    ) -> None:
        frame = df.reset_index(drop=True)
        self.ds = pd.to_datetime(frame["ds"])
        self.y = frame["y"].to_numpy(dtype=float)
        self.period, self.order = period, order
        self.engine_requested = engine
        self.engine = resolve_engine(engine)
        self.seed, self.draws, self.tune, self.chains = seed, draws, tune, chains
        self.levels = tuple(levels)
        # Retail demand is multiplicative and right-skewed, so the Gaussian
        # belongs on log(y), not on y. Exponentiating the predictive draws
        # leaves the median and every quantile exact (a monotone transform),
        # so no smearing correction is needed.
        self.log_target = log_target
        self._y_model = np.log(self.y) if log_target else self.y

        self._n = self.y.size
        self._t = np.arange(self._n, dtype=float)
        self._denom = max(self._n - 1, 1.0)
        self._holiday_columns: list[str] = []
        self._holidays = holiday_indicators(self.ds, holiday_dates)
        self._holiday_columns = list(self._holidays.columns)
        self.X, self.names = build_design(
            self._t, self._denom, period, order, self._holidays
        )
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self) -> "BayesianForecast":
        if self.engine == "pymc":
            self._fit_pymc()
        else:
            self._fit_closed_form()
        self._fitted = True
        return self

    def _fit_closed_form(self) -> None:
        n, p = self.X.shape
        y = self._y_model
        xtx_inv = np.linalg.pinv(self.X.T @ self.X)
        self.beta_hat = xtx_inv @ (self.X.T @ y)
        resid = y - self.X @ self.beta_hat
        self.nu = float(max(n - p, 1))
        self.sigma2_hat = float(resid @ resid) / self.nu
        self._xtx_inv = xtx_inv
        self._chol = np.linalg.cholesky(xtx_inv + 1e-12 * np.eye(p))

    def _fit_pymc(self) -> None:
        import pymc as pm

        y = self._y_model
        # Fit on a standardised response so the prior is genuinely weakly
        # informative and scale-free: N(0, 1) on z = (y - loc) / scale means the
        # same thing whether y is in units (millions) or in logs. On the level
        # scale a raw N(0, std(y)) prior would put the intercept ~170 prior SDs
        # away from its posterior mass and stall NUTS. Coefficients are mapped
        # back to the response scale afterwards, so the reported summary is
        # directly comparable with the closed-form engine.
        loc = float(np.mean(y))
        scale = float(np.std(y)) or 1.0
        z = (y - loc) / scale
        with pm.Model() as model:
            beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=len(self.names))
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            pm.Normal("obs", mu=self.X @ beta, sigma=sigma, observed=z)
            self.idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=0.9,
                random_seed=self.seed,
                progressbar=False,
            )
        beta_z = self.idata.posterior["beta"].values.reshape(-1, len(self.names))
        sigma_z = self.idata.posterior["sigma"].values.reshape(-1)
        # mu_y = loc + scale * X @ beta_z, and X's first column is the intercept.
        self._beta_draws = beta_z * scale
        self._beta_draws[:, 0] += loc
        self._sigma_draws = sigma_z * scale

    # ------------------------------------------------------------- predict
    def _future_design(self, periods: int) -> np.ndarray:
        t_future = self._n + np.arange(periods, dtype=float)
        # Future weeks are not known holidays -> zero dummies (same shape as fit).
        future_holidays = None
        if self._holiday_columns:
            future_holidays = pd.DataFrame(
                np.zeros((periods, len(self._holiday_columns))),
                columns=self._holiday_columns,
            )
        X, _ = build_design(t_future, self._denom, self.period, self.order, future_holidays)
        return X

    def sample_predictive(
        self, periods: int = 6, n_samples: int = 2000, seed: int | None = None
    ) -> tuple[pd.Series, np.ndarray]:
        """Draw posterior predictive samples: (dates, samples[n_periods, n_samples])."""
        if not self._fitted:
            raise RuntimeError("call .fit() before .sample_predictive()")
        X_future = self._future_design(periods)
        dates = future_dates(self.ds.iloc[-1], periods)
        rng = np.random.default_rng(self.seed if seed is None else seed)

        if self.engine == "pymc":
            available = self._beta_draws.shape[0]
            if n_samples < available:
                pick = rng.choice(available, size=n_samples, replace=False)
                beta_draws, sigma_draws = self._beta_draws[pick], self._sigma_draws[pick]
            else:
                beta_draws, sigma_draws = self._beta_draws, self._sigma_draws
        else:
            # InvGamma(a=nu/2, scale=nu*s2/2) via 1 / Gamma(a, scale=1/(nu*s2/2)).
            a = self.nu / 2.0
            b = self.nu * self.sigma2_hat / 2.0
            sigma2 = 1.0 / rng.gamma(shape=a, scale=1.0 / b, size=n_samples)
            z = rng.standard_normal((n_samples, self.X.shape[1]))
            beta_draws = self.beta_hat + (z @ self._chol.T) * np.sqrt(sigma2)[:, None]
            sigma_draws = np.sqrt(sigma2)

        mean_draws = X_future @ beta_draws.T  # (periods, n_samples)
        samples = mean_draws + rng.standard_normal(mean_draws.shape) * sigma_draws[None, :]
        if self.log_target:
            samples = np.exp(samples)  # exact under a monotone transform
        return dates, samples

    def predict(
        self,
        periods: int = 6,
        levels=None,
        n_samples: int = 2000,
    ) -> pd.DataFrame:
        """Posterior predictive median + central credible bands."""
        levels = tuple(levels or self.levels)
        dates, samples = self.sample_predictive(periods, n_samples)
        out = pd.DataFrame({"ds": dates, "yhat": np.median(samples, axis=1)})
        for level in levels:
            alpha = 1.0 - level
            tag = level_tag(level)
            # Demand cannot be negative: clip at zero (a physical constraint,
            # not a modelling trick) and keep the median un-clipped.
            out[f"lower_{tag}"] = np.maximum(
                np.quantile(samples, alpha / 2.0, axis=1), 0.0
            )
            out[f"upper_{tag}"] = np.quantile(samples, 1.0 - alpha / 2.0, axis=1)
        return out.loc[:, ["ds", "yhat", *band_columns(levels)]]

    def parameter_summary(self) -> pd.DataFrame:
        """Posterior mean / sd / 95% credible bounds for every regression term."""
        if not self._fitted:
            raise RuntimeError("call .fit() before .parameter_summary()")
        if self.engine == "pymc":
            draws = self._beta_draws
            mean, sd = draws.mean(axis=0), draws.std(axis=0, ddof=1)
            lo, hi = np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)
        else:
            mean = self.beta_hat
            sd = np.sqrt(self.sigma2_hat * np.diag(self._xtx_inv))
            lo, hi = mean - 1.96 * sd, mean + 1.96 * sd
        return pd.DataFrame(
            {"term": self.names, "mean": mean, "sd": sd, "lower_95": lo, "upper_95": hi}
        )


def bayesian_forecast(
    train_df: pd.DataFrame,
    periods: int = 6,
    levels=DEFAULT_LEVELS,
    **kwargs,
) -> pd.DataFrame:
    """Stateless convenience wrapper: fit and forecast in one call."""
    return BayesianForecast(train_df, levels=levels, **kwargs).fit().predict(
        periods=periods, levels=levels
    )
