"""
lead_lag_engine.py - 크로스에셋 리드-래그 분석 엔진

리드 자산(미국 주식 프록시)과 래그 자산(한국 주식) 간의
동적 리드-래그 관계를 Granger 인과성 + 롤링 상관관계로 추적하고,
미반영 다이버전스를 감지하여 매매 기회를 포착.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from collections import deque

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from .utils import setup_logger, calc_returns


@dataclass
class PairHealth:
    """리드-래그 페어 건강 상태"""
    pair_name: str
    granger_p_value: float
    granger_pass_rate: float
    rolling_correlation: float
    is_healthy: bool
    last_checked: datetime = field(default_factory=datetime.now)
    reason_if_unhealthy: str = ""


@dataclass
class PairAnalysis:
    """단일 페어 분석 결과"""
    pair_name: str
    lead_symbol: str
    lag_symbols: list[str]

    # 관계 강도
    granger_p_value: float
    rolling_correlation: float
    beta: float
    is_healthy: bool

    # 다이버전스
    lead_return_n_day: float
    expected_lag_return: float
    actual_lag_return: float
    divergence: float
    divergence_zscore: float
    price_in_ratio: float

    # 시그널 방향
    direction: str  # "long" or "short"

    @property
    def signal_strength(self) -> float:
        """원시 시그널 강도 [0, 1]"""
        if not self.is_healthy:
            return 0.0
        div_score = min(abs(self.divergence) / 0.05, 1.0)
        z_score = min(abs(self.divergence_zscore) / 3.0, 1.0)
        corr_score = min(abs(self.rolling_correlation), 1.0)
        return div_score * 0.4 + z_score * 0.35 + corr_score * 0.25


class LeadLagEngine:
    """크로스에셋 리드-래그 분석 엔진"""

    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("LeadLag")

        self.granger_max_lag = config.get("granger_max_lag", 5)
        self.granger_p_threshold = config.get("granger_p_value_threshold", 0.05)
        self.granger_window = config.get("granger_window", 60)
        self.granger_min_obs = config.get("granger_min_obs", 40)
        self.granger_recompute = config.get("granger_recompute_interval", 5)

        self.corr_window = config.get("correlation_window", 30)
        self.corr_min = config.get("correlation_min", 0.3)
        self.corr_lag = config.get("correlation_lag", 1)

        self.div_lookback = config.get("divergence_lookback", 5)
        self.min_lead_move = config.get("min_lead_move_pct", 0.02)
        self.max_price_in = config.get("max_price_in_ratio", 0.7)
        self.zscore_threshold = config.get("divergence_zscore_threshold", 1.5)

        self.pair_configs = config.get("pairs", [])

        self._pair_health: dict[str, PairHealth] = {}
        self._granger_cache: dict[str, tuple[float, int]] = {}
        self._beta_cache: dict[str, float] = {}
        self._divergence_history: dict[str, deque] = {}
        self._granger_compute_counter: dict[str, int] = {}

    def compute_all_pairs(
        self,
        lead_data: dict[str, pd.DataFrame],
        lag_data: dict[str, pd.DataFrame],
    ) -> list[PairAnalysis]:
        """모든 리드-래그 페어 분석"""
        results = []

        for pair_cfg in self.pair_configs:
            pair_name = pair_cfg["name"]
            lead_symbol = pair_cfg["lead"]["symbol"]
            lag_syms = [s["symbol"] for s in pair_cfg.get("lag", [])]
            default_beta = pair_cfg.get("default_beta", 1.0)

            if lead_symbol not in lead_data:
                continue

            lead_df = lead_data[lead_symbol]
            if lead_df is None or len(lead_df) < self.granger_min_obs:
                continue

            lag_prices_map = {}
            for sym in lag_syms:
                if sym in lag_data and lag_data[sym] is not None and len(lag_data[sym]) > 20:
                    lag_prices_map[sym] = lag_data[sym]

            if not lag_prices_map:
                continue

            analysis = self._compute_single_pair(
                pair_cfg, lead_df, lag_prices_map, default_beta
            )
            if analysis:
                results.append(analysis)

        return results

    def _compute_single_pair(
        self,
        pair_cfg: dict,
        lead_df: pd.DataFrame,
        lag_prices_map: dict[str, pd.DataFrame],
        default_beta: float,
    ) -> Optional[PairAnalysis]:
        """단일 페어 분석"""
        pair_name = pair_cfg["name"]
        lead_symbol = pair_cfg["lead"]["symbol"]
        lag_symbols = list(lag_prices_map.keys())

        try:
            # 날짜 기반 forward fill 정렬 (US→KR 시차 반영)
            aligned = self._align_by_date(lead_df, lag_prices_map[lag_symbols[0]])
            if aligned is None or len(aligned) < self.granger_min_obs:
                return None

            aligned_lead = aligned["lead_ret"]
            aligned_lag = aligned["lag_ret"]
            lead_prices_aligned = aligned["lead_close"]
            lag_prices_aligned = aligned["lag_close"]

            # 대표 래그 종목 (첫 번째)
            rep_lag_sym = lag_symbols[0]

            # Granger 인과성
            granger_p, _ = self._test_granger(pair_name, aligned_lead, aligned_lag)

            # 롤링 상관관계 (리드[t] vs 래그[t+1])
            corr = self._compute_rolling_correlation(aligned_lead, aligned_lag)

            # 베타 추정
            beta = self._compute_beta(pair_name, aligned_lead, aligned_lag, default_beta)

            # 페어 건강도: Granger OR 상관관계 통과 시 healthy
            is_healthy = (
                granger_p < self.granger_p_threshold
                or abs(corr) >= self.corr_min
            )
            self._pair_health[pair_name] = PairHealth(
                pair_name=pair_name,
                granger_p_value=granger_p,
                granger_pass_rate=1.0 if granger_p < self.granger_p_threshold else 0.0,
                rolling_correlation=corr,
                is_healthy=is_healthy,
                reason_if_unhealthy="" if is_healthy else f"Granger p={granger_p:.3f}, corr={corr:.3f}",
            )

            # 다이버전스 계산 (원본 가격으로 - ffill 왜곡 방지)
            lead_prices_orig = lead_df["close"].reset_index(drop=True)
            rep_lag_df = lag_prices_map[rep_lag_sym]
            lag_prices_orig = rep_lag_df["close"].reset_index(drop=True)

            div_result = self._compute_divergence(
                lead_prices_orig, lag_prices_orig, beta
            )
            if div_result is None:
                return None

            lead_ret, expected, actual, divergence, price_in = div_result

            # 최소 움직임 필터
            if abs(lead_ret) < self.min_lead_move:
                return None

            # 이미 반영된 경우 스킵
            if price_in > self.max_price_in:
                return None

            # z-score
            zscore = self._compute_divergence_zscore(pair_name, divergence)

            # 방향 결정
            direction = "long" if divergence > 0 else "short"

            return PairAnalysis(
                pair_name=pair_name,
                lead_symbol=lead_symbol,
                lag_symbols=lag_symbols,
                granger_p_value=granger_p,
                rolling_correlation=corr,
                beta=beta,
                is_healthy=is_healthy,
                lead_return_n_day=lead_ret,
                expected_lag_return=expected,
                actual_lag_return=actual,
                divergence=divergence,
                divergence_zscore=zscore,
                price_in_ratio=price_in,
                direction=direction,
            )

        except Exception as e:
            self.logger.error(f"페어 분석 실패 [{pair_name}]: {e}")
            return None

    # ──────────────────────────────────────────────
    # Granger 인과성
    # ──────────────────────────────────────────────

    def _test_granger(
        self,
        pair_name: str,
        lead_returns: pd.Series,
        lag_returns: pd.Series,
    ) -> tuple[float, int]:
        """Granger 인과성 테스트 (캐시 + 재계산 간격 관리)"""
        counter = self._granger_compute_counter.get(pair_name, 0) + 1
        self._granger_compute_counter[pair_name] = counter

        if pair_name in self._granger_cache and counter % self.granger_recompute != 0:
            return self._granger_cache[pair_name]

        result = self._run_granger_test(lead_returns, lag_returns)
        self._granger_cache[pair_name] = result
        return result

    def _run_granger_test(
        self,
        lead_returns: pd.Series,
        lag_returns: pd.Series,
    ) -> tuple[float, int]:
        """statsmodels Granger 인과성 테스트 실행"""
        try:
            # lead[t-1]이 lag[t]를 예측하는지 테스트
            lead_shifted = lead_returns.shift(1)
            data = pd.DataFrame({
                "lag": lag_returns,
                "lead": lead_shifted,
            }).dropna()

            if len(data) < self.granger_min_obs:
                return 1.0, 0

            max_lag = min(self.granger_max_lag, len(data) // 10)
            if max_lag < 1:
                return 1.0, 0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                results = grangercausalitytests(data.values, maxlag=max_lag, verbose=False)

            min_p = 1.0
            best_lag = 1
            for lag_val in range(1, max_lag + 1):
                p_value = results[lag_val][0]["ssr_ftest"][1]
                if p_value < min_p:
                    min_p = p_value
                    best_lag = lag_val

            return min_p, best_lag

        except Exception:
            return 1.0, 0

    # ──────────────────────────────────────────────
    # 상관관계 & 베타
    # ──────────────────────────────────────────────

    def _align_by_date(
        self,
        lead_df: pd.DataFrame,
        lag_df: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """날짜 기반 forward fill 정렬 (US 데이터를 KR 날짜에 매핑)"""
        try:
            lead = lead_df[["date", "close"]].copy()
            lag = lag_df[["date", "close"]].copy()

            lead["date"] = pd.to_datetime(lead["date"])
            lag["date"] = pd.to_datetime(lag["date"])

            lead_indexed = lead.set_index("date").rename(columns={"close": "lead_close"})
            lag_indexed = lag.set_index("date").rename(columns={"close": "lag_close"})

            # KR 날짜 기준으로 left join, US 데이터를 forward fill
            combined = lag_indexed.join(lead_indexed, how="left")
            combined["lead_close"] = combined["lead_close"].ffill()
            combined = combined.dropna()

            if len(combined) < 20:
                return None

            combined["lead_ret"] = combined["lead_close"].pct_change()
            combined["lag_ret"] = combined["lag_close"].pct_change()
            combined = combined.dropna()

            # 인덱스를 정수로 리셋
            combined = combined.reset_index(drop=True)
            return combined

        except Exception:
            return None

    def _compute_rolling_correlation(
        self,
        lead_returns: pd.Series,
        lag_returns: pd.Series,
    ) -> float:
        """리드[t]와 래그[t+1]의 롤링 상관관계"""
        lead_shifted = lead_returns.shift(self.corr_lag)
        combined = pd.DataFrame({
            "lead": lead_shifted,
            "lag": lag_returns,
        }).dropna()

        if len(combined) < self.corr_window:
            return 0.0

        recent = combined.iloc[-self.corr_window:]
        corr = recent["lead"].corr(recent["lag"])
        return float(corr) if not pd.isna(corr) else 0.0

    def _compute_beta(
        self,
        pair_name: str,
        lead_returns: pd.Series,
        lag_returns: pd.Series,
        default_beta: float,
    ) -> float:
        """회귀 베타: lag[t] = alpha + beta * lead[t-1]"""
        try:
            lead_shifted = lead_returns.shift(1)
            combined = pd.DataFrame({
                "lead": lead_shifted,
                "lag": lag_returns,
            }).dropna()

            window = min(self.granger_window, len(combined))
            if window < 20:
                return default_beta

            recent = combined.iloc[-window:]
            beta = np.polyfit(recent["lead"].values, recent["lag"].values, 1)[0]
            beta = float(np.clip(beta, -5.0, 5.0))
            self._beta_cache[pair_name] = beta
            return beta

        except Exception:
            return self._beta_cache.get(pair_name, default_beta)

    # ──────────────────────────────────────────────
    # 다이버전스
    # ──────────────────────────────────────────────

    def _compute_divergence(
        self,
        lead_prices: pd.Series,
        lag_prices: pd.Series,
        beta: float,
    ) -> Optional[tuple[float, float, float, float, float]]:
        """리드-래그 다이버전스 계산"""
        n = self.div_lookback

        if len(lead_prices) < n + 1 or len(lag_prices) < n + 1:
            return None

        lead_return = (lead_prices.iloc[-1] / lead_prices.iloc[-n - 1]) - 1
        expected_lag = beta * lead_return
        actual_lag = (lag_prices.iloc[-1] / lag_prices.iloc[-n - 1]) - 1

        if abs(expected_lag) < 0.001:
            return None

        divergence = expected_lag - actual_lag

        price_in = abs(actual_lag / expected_lag) if expected_lag != 0 else 1.0
        price_in = min(price_in, 2.0)

        return lead_return, expected_lag, actual_lag, divergence, price_in

    def _compute_divergence_zscore(
        self, pair_name: str, current_divergence: float
    ) -> float:
        """다이버전스 z-score 계산"""
        if pair_name not in self._divergence_history:
            self._divergence_history[pair_name] = deque(maxlen=60)

        history = self._divergence_history[pair_name]
        history.append(current_divergence)

        if len(history) < 10:
            return abs(current_divergence) / 0.02 if current_divergence != 0 else 0.0

        arr = np.array(history)
        mean = arr.mean()
        std = arr.std()

        if std < 1e-8:
            return 0.0

        return float((current_divergence - mean) / std)

    # ──────────────────────────────────────────────
    # 페어 건강도
    # ──────────────────────────────────────────────

    def get_all_health(self) -> dict[str, PairHealth]:
        return self._pair_health.copy()

    def get_healthy_pairs(self) -> list[str]:
        return [name for name, h in self._pair_health.items() if h.is_healthy]
