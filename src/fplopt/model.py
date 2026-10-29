"""Two-stage expected-points model with calibrated prediction intervals.

    E[points] = P(1-59 min) * E[pts | 1-59] + P(60+ min) * E[pts | 60+]      (0 minutes -> 0 points)

This is the same decomposition as credit's EL = PD x LGD x EAD: one model for *whether* the
player plays, another for *how much* he scores when he does. Around 60% of player-gameweeks
are zero-minute rows, so a single regressor has to average two very different populations.

Intervals: direct quantile gradient boosting (10% and 90%), then a conformalised quantile
regression (CQR) correction computed from recent out-of-sample errors, so the 80% interval
is calibrated on data the model has not seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

GB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                 l2_regularization=1.0, random_state=42)


def _cat_mask(cols: list[str]) -> list[bool]:
    return [c == "pos_code" for c in cols]


def _usable(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Drop columns with fewer than two distinct values in this training window (e.g. a regime flag
    or a statistic that did not exist yet); gradient boosting cannot bin them."""
    return [c for c in cols if df[c].nunique(dropna=True) > 1]


@dataclass
class TwoStageModel:
    features: list[str]
    alpha: float = 0.2
    clf: HistGradientBoostingClassifier | None = field(default=None, repr=False)
    reg: dict = field(default_factory=dict, repr=False)
    q_lo: HistGradientBoostingRegressor | None = field(default=None, repr=False)
    q_hi: HistGradientBoostingRegressor | None = field(default=None, repr=False)

    def fit(self, df: pd.DataFrame) -> TwoStageModel:
        self.features = _usable(df, self.features)
        X = df[self.features]
        cat = _cat_mask(self.features)
        self.clf = HistGradientBoostingClassifier(categorical_features=cat, **GB_PARAMS)
        self.clf.fit(X, df["minutes_bucket"])
        for b in (1, 2):
            m = df["minutes_bucket"] == b
            r = HistGradientBoostingRegressor(categorical_features=cat, **GB_PARAMS)
            self.reg[b] = r.fit(X[m], df.loc[m, "total_points"])
        lo, hi = self.alpha / 2, 1 - self.alpha / 2
        self.q_lo = HistGradientBoostingRegressor(loss="quantile", quantile=lo, categorical_features=cat, **GB_PARAMS)
        self.q_hi = HistGradientBoostingRegressor(loss="quantile", quantile=hi, categorical_features=cat, **GB_PARAMS)
        self.q_lo.fit(X, df["total_points"])
        self.q_hi.fit(X, df["total_points"])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.clf is None:
            raise RuntimeError("model is not fitted")
        X = df[self.features]
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        p = {b: proba[:, classes.index(b)] if b in classes else np.zeros(len(df)) for b in (0, 1, 2)}
        m1 = self.reg[1].predict(X)
        m2 = self.reg[2].predict(X)
        mu = p[1] * m1 + p[2] * m2
        out = pd.DataFrame({
            "p_play0": p[0], "p_play1_59": p[1], "p_play60": p[2],
            "pts_if_1_59": m1, "pts_if_60": m2, "mu": mu,
            "q_lo_raw": self.q_lo.predict(X), "q_hi_raw": self.q_hi.predict(X),
        }, index=df.index)
        return out


@dataclass
class SingleModel:
    """Ablation: one gradient-boosting regressor on total points (no play/score split)."""
    features: list[str]
    reg: HistGradientBoostingRegressor | None = field(default=None, repr=False)

    def fit(self, df: pd.DataFrame) -> SingleModel:
        self.features = _usable(df, self.features)
        self.reg = HistGradientBoostingRegressor(categorical_features=_cat_mask(self.features), **GB_PARAMS)
        self.reg.fit(df[self.features], df["total_points"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(df[self.features])


def cqr_correction(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.2) -> float:
    """Conformalised quantile regression: the amount q to widen [lo, hi] by for 1-alpha coverage.

    Scores s_i = max(lo_i - y_i, y_i - hi_i); q = the ceil((n+1)(1-alpha))-th smallest score.
    """
    y, lo, hi = map(np.asarray, (y, lo, hi))
    if len(y) == 0:
        return 0.0
    s = np.maximum(lo - y, y - hi)
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(np.sort(s)[min(k, len(s)) - 1])
