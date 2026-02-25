"""
regime_detector.py - 복합 레짐 감지 엔진

변동성 기반 1차 분류 + HMM 보조 확인으로 시장 상태를 판별:
  - TREND: 저변동성, 추세 시장 → 리드-래그 풀 가동
  - VOLATILE: 고변동성, 평균회귀 → 리드-래그 + MR 필터
  - CRASH: 극단 변동성 → 전략 차단, 현금 확보
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from .utils import (
    setup_logger, calc_returns, calc_realized_volatility,
    calc_vol_of_vol, calc_sma,
)


class Regime(Enum):
    TREND = "TREND"
    VOLATILE = "VOLATILE"
    CRASH = "CRASH"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeState:
    regime: Regime
    confidence: float
    duration: int
    just_switched: bool
    prev_regime: Optional[Regime] = None
    vol_short: float = 0.0
    vol_long: float = 0.0
    vol_of_vol: float = 0.0
    trend_direction: str = "flat"
    regime_rules: dict = field(default_factory=dict)

    @property
    def position_multiplier(self) -> float:
        return self.regime_rules.get("position_size_multiplier", 0.0)

    @property
    def allowed_strategies(self) -> list[str]:
        return self.regime_rules.get("allowed_strategies", [])

    @property
    def atr_stop_multiplier(self) -> float:
        return self.regime_rules.get("atr_stop_multiplier", 2.0)

    @property
    def max_exposure(self) -> float:
        return self.regime_rules.get("max_exposure", 0.80)


class CompositeRegimeDetector:
    """변동성 기반 + HMM 보조 확인 복합 레짐 감지기"""

    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("RegimeDetector")

        self.vol_short_window = config.get("vol_short_window", 5)
        self.vol_long_window = config.get("vol_long_window", 20)
        self.vov_window = config.get("vol_of_vol_window", 20)

        self.trend_vol_upper = config.get("trend_vol_upper", 0.15)
        self.volatile_vol_upper = config.get("volatile_vol_upper", 0.35)
        self.vov_crash_threshold = config.get("vov_crash_threshold", 0.5)
        self.vov_trend_threshold = config.get("vov_trend_threshold", 0.2)

        self.sma_short = config.get("sma_short", 20)
        self.sma_long = config.get("sma_long", 60)

        self.hmm_enabled = config.get("hmm_enabled", True)
        self.hmm_states = config.get("hmm_states", 3)
        self.hmm_lookback = config.get("hmm_lookback_days", 90)
        self.hmm_retrain_interval = config.get("hmm_retrain_interval", 14)
        self.hmm_weight = config.get("hmm_weight", 0.3)

        self.min_regime_days = config.get("min_regime_days", 2)

        self.regime_rules = {
            Regime.TREND: config.get("TREND", {}),
            Regime.VOLATILE: config.get("VOLATILE", {}),
            Regime.CRASH: config.get("CRASH", {}),
            Regime.UNKNOWN: config.get("VOLATILE", {}),
        }

        self._hmm_model: Optional[GaussianHMM] = None
        self._hmm_state_mapping: dict[int, Regime] = {}
        self._last_train_date: Optional[datetime] = None

        self._confirmed_regime = Regime.UNKNOWN
        self._pending_regime = Regime.UNKNOWN
        self._pending_count = 0
        self._duration = 0

    def detect(self, prices: pd.Series) -> RegimeState:
        """레짐 감지 메인 파이프라인"""
        if len(prices) < self.vol_long_window + 5:
            return self._make_unknown_state()

        vol_regime, vol_conf, vol_metrics = self._classify_by_volatility(prices)

        hmm_result = None
        if self.hmm_enabled and self._hmm_model is not None:
            hmm_result = self._classify_by_hmm(prices)

        final_regime, final_conf = self._merge_classifications(
            (vol_regime, vol_conf, vol_metrics), hmm_result
        )

        prev_regime = self._confirmed_regime
        just_switched = False
        confirmed = self._apply_stability_filter(final_regime)

        if confirmed != prev_regime and prev_regime != Regime.UNKNOWN:
            just_switched = True
            self._duration = 1
            self.logger.info(
                f"레짐 전환: {prev_regime.value} → {confirmed.value} "
                f"(확신도: {final_conf:.1%})"
            )
        else:
            self._duration += 1

        trend_dir = self._detect_trend_direction(prices)

        return RegimeState(
            regime=confirmed,
            confidence=final_conf,
            duration=self._duration,
            just_switched=just_switched,
            prev_regime=prev_regime if just_switched else None,
            vol_short=vol_metrics.get("vol_short", 0),
            vol_long=vol_metrics.get("vol_long", 0),
            vol_of_vol=vol_metrics.get("vov", 0),
            trend_direction=trend_dir,
            regime_rules=self.regime_rules.get(confirmed, {}),
        )

    def _classify_by_volatility(
        self, prices: pd.Series
    ) -> tuple[Regime, float, dict]:
        """1차: 실현 변동성 기반 분류"""
        returns = calc_returns(prices)

        vol_short = float(returns.iloc[-self.vol_short_window:].std() * np.sqrt(252))
        vol_long = float(returns.iloc[-self.vol_long_window:].std() * np.sqrt(252))
        vov = calc_vol_of_vol(prices, self.vol_long_window, self.vov_window)

        metrics = {"vol_short": vol_short, "vol_long": vol_long, "vov": vov}

        # CRASH: 극단 변동성
        if vol_long > self.volatile_vol_upper:
            conf = min(0.7 + (vol_long - self.volatile_vol_upper) * 2, 1.0)
            return Regime.CRASH, conf, metrics

        if vol_long > self.volatile_vol_upper * 0.85 and vov > self.vov_crash_threshold:
            conf = min(0.6 + vov * 0.3, 0.95)
            return Regime.CRASH, conf, metrics

        # 급변 에스컬레이션: 단기 변동성이 장기의 2.5배 초과 + 장기도 높아야
        if vol_short > vol_long * 2.5 and vol_long > self.volatile_vol_upper * 0.85:
            conf = min(0.6 + (vol_short / vol_long - 2.5) * 0.5, 0.9)
            return Regime.CRASH, conf, metrics

        # VOLATILE: 중간 변동성
        if vol_long > self.trend_vol_upper:
            distance = (vol_long - self.trend_vol_upper) / (
                self.volatile_vol_upper - self.trend_vol_upper
            )
            conf = 0.6 + distance * 0.3
            return Regime.VOLATILE, min(conf, 0.95), metrics

        # TREND: 저변동성
        conf = 0.7
        if vov < self.vov_trend_threshold:
            conf = 0.85
        if vol_long < self.trend_vol_upper * 0.7:
            conf = 0.95
        return Regime.TREND, conf, metrics

    def _classify_by_hmm(
        self, prices: pd.Series
    ) -> Optional[tuple[Regime, float]]:
        """2차: HMM 보조 확인"""
        try:
            features = self._extract_hmm_features(prices)
            if features is None or len(features) < 10:
                return None

            state_probs = self._hmm_model.predict_proba(features)
            current_state = self._hmm_model.predict(features)[-1]
            confidence = float(state_probs[-1][current_state])

            regime = self._hmm_state_mapping.get(current_state, Regime.UNKNOWN)
            return regime, confidence

        except Exception as e:
            self.logger.warning(f"HMM 분류 실패: {e}")
            return None

    def _merge_classifications(
        self,
        vol_result: tuple[Regime, float, dict],
        hmm_result: Optional[tuple[Regime, float]],
    ) -> tuple[Regime, float]:
        """변동성 + HMM 결과 합산"""
        vol_regime, vol_conf, _ = vol_result

        if hmm_result is None:
            return vol_regime, vol_conf

        hmm_regime, hmm_conf = hmm_result

        if vol_regime == hmm_regime:
            return vol_regime, max(vol_conf, hmm_conf)

        # HMM CRASH 에스컬레이션 비활성 (변동성 분류가 우선)
        # 변동성이 항상 우선, 확신도만 감소
        blended_conf = vol_conf * (1 - self.hmm_weight) + hmm_conf * self.hmm_weight
        return vol_regime, blended_conf * 0.85

    def _apply_stability_filter(self, regime: Regime) -> Regime:
        """히스테리시스: min_regime_days 연속 유지 시에만 전환"""
        if regime == self._confirmed_regime:
            self._pending_regime = regime
            self._pending_count = 0
            return self._confirmed_regime

        # UNKNOWN에서 첫 전환 / CRASH는 즉시 전환 (안전 우선)
        if self._confirmed_regime == Regime.UNKNOWN or regime == Regime.CRASH:
            self._confirmed_regime = regime
            self._pending_regime = regime
            self._pending_count = 0
            return regime

        if regime == self._pending_regime:
            self._pending_count += 1
        else:
            self._pending_regime = regime
            self._pending_count = 1

        if self._pending_count >= self.min_regime_days:
            self._confirmed_regime = regime
            self._pending_count = 0

        return self._confirmed_regime

    def _detect_trend_direction(self, prices: pd.Series) -> str:
        """SMA 기반 추세 방향 판별"""
        if len(prices) < self.sma_long + 5:
            return "flat"

        sma_s = calc_sma(prices, self.sma_short)
        sma_l = calc_sma(prices, self.sma_long)

        sma_s_val = sma_s.iloc[-1]
        sma_l_val = sma_l.iloc[-1]
        current = prices.iloc[-1]

        if pd.isna(sma_s_val) or pd.isna(sma_l_val):
            return "flat"

        if sma_s_val > sma_l_val and current > sma_s_val:
            return "up"
        elif sma_s_val < sma_l_val and current < sma_s_val:
            return "down"
        return "flat"

    # ──────────────────────────────────────────────
    # HMM 학습
    # ──────────────────────────────────────────────

    def fit_hmm(self, prices: pd.Series) -> bool:
        """HMM 모델 학습"""
        if not self.hmm_enabled:
            return False

        try:
            features = self._extract_hmm_features(prices)
            if features is None or len(features) < 30:
                self.logger.warning("HMM 학습 데이터 부족")
                return False

            self._hmm_model = GaussianHMM(
                n_components=self.hmm_states,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                tol=0.01,
            )
            self._hmm_model.fit(features)
            self._map_hmm_states(features)
            self._last_train_date = datetime.now()
            self.logger.info("HMM 모델 학습 완료")
            return True

        except Exception as e:
            self.logger.error(f"HMM 학습 실패: {e}")
            return False

    def needs_retrain(self) -> bool:
        if self._last_train_date is None:
            return True
        return (datetime.now() - self._last_train_date).days >= self.hmm_retrain_interval

    def _extract_hmm_features(self, prices: pd.Series) -> Optional[np.ndarray]:
        """HMM 입력 특성: 수익률, 변동성, 자기상관"""
        if len(prices) < 20:
            return None

        returns = calc_returns(prices)
        vol_5d = returns.rolling(5).std()
        autocorr = returns.rolling(10).apply(
            lambda x: x.autocorr(lag=1) if len(x) >= 5 else 0,
            raw=False,
        )

        df = pd.DataFrame({
            "returns": returns,
            "volatility": vol_5d,
            "autocorr": autocorr,
        }).dropna()

        if len(df) < 10:
            return None
        return df.values

    def _map_hmm_states(self, features: np.ndarray):
        """HMM 상태를 변동성 기반 레짐으로 매핑"""
        states = self._hmm_model.predict(features)
        state_vols = {}
        for s in range(self.hmm_states):
            mask = states == s
            if mask.sum() > 0:
                state_vols[s] = features[mask, 1].mean()  # col 1 = volatility
            else:
                state_vols[s] = 0

        sorted_states = sorted(state_vols.items(), key=lambda x: x[1])

        if self.hmm_states == 3:
            self._hmm_state_mapping = {
                sorted_states[0][0]: Regime.TREND,
                sorted_states[1][0]: Regime.VOLATILE,
                sorted_states[2][0]: Regime.CRASH,
            }
        elif self.hmm_states == 2:
            self._hmm_state_mapping = {
                sorted_states[0][0]: Regime.TREND,
                sorted_states[1][0]: Regime.VOLATILE,
            }

    def _make_unknown_state(self) -> RegimeState:
        return RegimeState(
            regime=Regime.UNKNOWN,
            confidence=0.0,
            duration=0,
            just_switched=False,
            regime_rules=self.regime_rules.get(Regime.UNKNOWN, {}),
        )

    def get_regime_summary(self, prices: pd.Series) -> dict:
        state = self.detect(prices)
        return {
            "regime": state.regime.value,
            "confidence": f"{state.confidence:.1%}",
            "duration_days": state.duration,
            "vol_short": f"{state.vol_short:.1%}",
            "vol_long": f"{state.vol_long:.1%}",
            "vol_of_vol": f"{state.vol_of_vol:.2f}",
            "trend_direction": state.trend_direction,
        }
