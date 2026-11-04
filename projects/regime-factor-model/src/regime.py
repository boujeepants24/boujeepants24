"""
regime.py
---------
Hidden Markov Model-based market regime detection.

Implemented from scratch using NumPy — no external HMM libraries required.

Model: Gaussian HMM with 3 hidden states fit via the Baum-Welch algorithm
(Expectation-Maximization). Decoding uses the Viterbi algorithm.

Observation features (computed from SPY returns):
  f1 : 21-day rolling mean return    (trend signal)
  f2 : 21-day rolling std of returns (short-term volatility)
  f3 : 63-day rolling cumulative log return (medium-term trend)

States are auto-labeled as Bull / Bear / High-Vol by inspecting the
emission means of each fitted state.

Factor weight sets per regime:
  Bull     : Momentum-heavy (trend following works in rising markets)
  Bear     : Quality + Low-Vol (flight to quality / defensive)
  High-Vol : Low-Vol dominant (volatility clustering = vol premium)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# ---------------------------------------------------------------------------
# Regime labels
# ---------------------------------------------------------------------------

REGIME_BULL     = 0
REGIME_BEAR     = 1
REGIME_HIGH_VOL = 2

REGIME_NAMES = {
    REGIME_BULL:     "Bull",
    REGIME_BEAR:     "Bear",
    REGIME_HIGH_VOL: "High-Vol",
}

# Factor weight sets per regime: [momentum, quality, low_vol, value]
REGIME_FACTOR_WEIGHTS = {
    REGIME_BULL:     np.array([0.40, 0.25, 0.15, 0.20]),
    REGIME_BEAR:     np.array([0.10, 0.35, 0.35, 0.20]),
    REGIME_HIGH_VOL: np.array([0.15, 0.20, 0.50, 0.15]),
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_hmm_features(bench_returns: pd.Series):
    """
    Build the 3-feature observation matrix for the HMM.
    Returns (ndarray of shape (T,3), DatetimeIndex).
    """
    r = bench_returns.copy()
    f1 = r.rolling(21).mean()
    f2 = r.rolling(21).std(ddof=1)
    f3 = r.rolling(63).sum()

    features = pd.concat([f1, f2, f3], axis=1)
    features.columns = ["mean21", "std21", "cum63"]
    features.dropna(inplace=True)
    return features.values, features.index


# ---------------------------------------------------------------------------
# Gaussian HMM — implemented from scratch
# ---------------------------------------------------------------------------

class GaussianHMM:
    """
    Gaussian Hidden Markov Model trained with Baum-Welch (EM).

    State space model:
      - K hidden states, each with a Gaussian emission N(mu_k, Sigma_k)
      - Transition matrix A[i,j] = P(z_t=j | z_{t-1}=i)
      - Initial state distribution pi[k] = P(z_1=k)

    Training: Baum-Welch (E-step: forward-backward, M-step: re-estimate params)
    Decoding: Viterbi algorithm (max-product)
    """

    def __init__(self, n_components: int = 3, n_iter: int = 100,
                 tol: float = 1e-4, random_state: int = 42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

        self.pi_: np.ndarray | None = None   # (K,)
        self.A_: np.ndarray | None = None    # (K, K)
        self.means_: np.ndarray | None = None  # (K, D)
        self.covars_: np.ndarray | None = None  # (K, D, D)

    # ------------------------------------------------------------------ init
    def _init_params(self, X: np.ndarray):
        K, D = self.n_components, X.shape[1]
        # K-means++ style init for means
        idx = self.rng.choice(len(X), K, replace=False)
        self.means_ = X[idx].copy().astype(float)
        self.covars_ = np.array([np.eye(D) * X.var(axis=0) for _ in range(K)])
        self.A_ = np.ones((K, K)) / K
        self.pi_ = np.ones(K) / K

    # --------------------------------------------------------- emission probs
    def _emission(self, X: np.ndarray) -> np.ndarray:
        """
        B[t, k] = p(x_t | z_t = k)  — shape (T, K)
        Uses diagonal Gaussian for numerical stability.
        """
        T, K = len(X), self.n_components
        B = np.zeros((T, K))
        for k in range(K):
            B[:, k] = multivariate_normal.pdf(
                X, mean=self.means_[k], cov=self.covars_[k], allow_singular=True
            )
        B = np.clip(B, 1e-300, None)
        return B

    # ------------------------------------------------------- forward algorithm
    def _forward(self, B: np.ndarray):
        """
        Log-scaled forward pass.
        alpha[t, k] = P(x_1..x_t, z_t=k)
        Returns alpha (T, K) and log-likelihood.
        """
        T, K = B.shape
        alpha = np.zeros((T, K))
        alpha[0] = self.pi_ * B[0]
        scale = np.zeros(T)
        scale[0] = alpha[0].sum()
        alpha[0] /= (scale[0] + 1e-300)

        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A_) * B[t]
            scale[t] = alpha[t].sum()
            alpha[t] /= (scale[t] + 1e-300)

        log_likelihood = np.sum(np.log(scale + 1e-300))
        return alpha, scale, log_likelihood

    # ------------------------------------------------------ backward algorithm
    def _backward(self, B: np.ndarray, scale: np.ndarray):
        """
        Scaled backward pass.
        beta[t, k] = P(x_{t+1}..x_T | z_t=k)
        """
        T, K = B.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0

        for t in range(T-2, -1, -1):
            beta[t] = (self.A_ @ (B[t+1] * beta[t+1]))
            beta[t] /= (scale[t+1] + 1e-300)

        return beta

    # ------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """Baum-Welch EM training."""
        self._init_params(X)
        T, D = X.shape
        K = self.n_components
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            B = self._emission(X)
            alpha, scale, log_ll = self._forward(B)
            beta = self._backward(B, scale)

            # E-step: compute gamma and xi
            # gamma[t, k] = P(z_t=k | X)
            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)

            # xi[t, i, j] = P(z_t=i, z_{t+1}=j | X)
            xi = np.zeros((T-1, K, K))
            for t in range(T-1):
                xi[t] = (alpha[t:t+1].T * self.A_ * (B[t+1] * beta[t+1]))
                xi[t] /= (xi[t].sum() + 1e-300)

            # M-step: update parameters
            self.pi_ = gamma[0] / (gamma[0].sum() + 1e-300)

            xi_sum = xi.sum(axis=0)
            self.A_ = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)

            gamma_sum = gamma.sum(axis=0)  # (K,)
            for k in range(K):
                self.means_[k] = (gamma[:, k:k+1] * X).sum(axis=0) / (gamma_sum[k] + 1e-300)
                diff = X - self.means_[k]
                cov = (gamma[:, k:k+1] * diff).T @ diff / (gamma_sum[k] + 1e-300)
                # Diagonal covariance + regularization
                self.covars_[k] = np.diag(np.diag(cov) + 1e-6)

            if abs(log_ll - prev_ll) < self.tol:
                break
            prev_ll = log_ll

        return self

    # --------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding — returns most likely state sequence."""
        T, K = len(X), self.n_components
        B = self._emission(X)

        viterbi = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)

        viterbi[0] = np.log(self.pi_ + 1e-300) + np.log(B[0] + 1e-300)

        log_A = np.log(self.A_ + 1e-300)
        for t in range(1, T):
            trans = viterbi[t-1:t].T + log_A
            psi[t] = trans.argmax(axis=0)
            viterbi[t] = trans.max(axis=0) + np.log(B[t] + 1e-300)

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = viterbi[-1].argmax()
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return smoothed posterior state probabilities (T, K)."""
        B = self._emission(X)
        alpha, scale, _ = self._forward(B)
        beta = self._backward(B, scale)
        gamma = alpha * beta
        gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)
        return gamma


# ---------------------------------------------------------------------------
# Regime detector wrapper
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Wraps the from-scratch GaussianHMM with:
      - Automatic state labeling (Bull / Bear / High-Vol)
      - Walk-forward predict interface
    """

    def __init__(self, n_regimes: int = 3, n_iter: int = 100,
                 random_state: int = 42):
        self.n_regimes    = n_regimes
        self.n_iter       = n_iter
        self.random_state = random_state
        self.model: GaussianHMM | None = None
        self._state_map: dict[int, int] = {}

    def fit(self, bench_returns: pd.Series) -> "RegimeDetector":
        features, _ = build_hmm_features(bench_returns)
        if len(features) < 100:
            raise ValueError(f"Not enough data: {len(features)} obs, need >= 100")

        model = GaussianHMM(
            n_components=self.n_regimes,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(features)
        self.model = model

        # Label states by emission means
        means = model.means_       # (K, 3)
        mean_ret = means[:, 0]
        mean_vol = means[:, 1]

        bull_raw = int(np.argmax(mean_ret))
        vols_excl = mean_vol.copy()
        vols_excl[bull_raw] = -np.inf
        high_vol_raw = int(np.argmax(vols_excl))
        remaining = [i for i in range(self.n_regimes)
                     if i not in (bull_raw, high_vol_raw)]
        bear_raw = remaining[0] if remaining else high_vol_raw

        self._state_map = {
            bull_raw:     REGIME_BULL,
            bear_raw:     REGIME_BEAR,
            high_vol_raw: REGIME_HIGH_VOL,
        }
        return self

    def predict(self, bench_returns: pd.Series) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Call fit() before predict()")
        features, idx = build_hmm_features(bench_returns)
        raw_states = self.model.predict(features)
        labeled = np.vectorize(self._state_map.get)(raw_states)
        return pd.Series(labeled, index=idx, name="regime")

    def predict_proba(self, bench_returns: pd.Series) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call fit() before predict()")
        features, idx = build_hmm_features(bench_returns)
        posteriors = self.model.predict_proba(features)
        col_map = {raw: REGIME_NAMES[lab] for raw, lab in self._state_map.items()}
        cols = [col_map.get(i, f"State{i}") for i in range(self.n_regimes)]
        return pd.DataFrame(posteriors, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# Regime-conditional factor weight optimizer
# ---------------------------------------------------------------------------

def optimize_regime_weights(
    regime_series: pd.Series,
    factor_return_df: pd.DataFrame,
) -> dict[int, np.ndarray]:
    from src.factors import FACTOR_NAMES
    optimized = {}
    for regime_id in [REGIME_BULL, REGIME_BEAR, REGIME_HIGH_VOL]:
        regime_dates = regime_series[regime_series == regime_id].index
        subset = factor_return_df.reindex(regime_dates).dropna()
        if len(subset) < 6:
            optimized[regime_id] = REGIME_FACTOR_WEIGHTS[regime_id].copy()
            continue
        sharpe = subset.mean() / (subset.std(ddof=1) + 1e-8)
        sharpe = sharpe.clip(lower=0)
        total = sharpe.sum()
        if total < 1e-8:
            optimized[regime_id] = REGIME_FACTOR_WEIGHTS[regime_id].copy()
        else:
            raw_weights = sharpe.values / total
            prior = REGIME_FACTOR_WEIGHTS[regime_id]
            blended = 0.5 * raw_weights + 0.5 * prior
            optimized[regime_id] = blended / blended.sum()
    return optimized


def get_regime_at_date(regime_series: pd.Series, date: pd.Timestamp) -> int:
    prior = regime_series[regime_series.index <= date]
    if prior.empty:
        return REGIME_BULL
    return int(prior.iloc[-1])
