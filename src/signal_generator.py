"""
signal_generator.py - Gap Trading Signal Generator

US 전일 종가 수익률 기반 한국장 시가 갭 트레이딩 시그널 생성.

로직:
  1. US 수익률 > threshold → 매수 시그널
  2. 시그널 강도 = US 수익률 크기에 비례
  3. 레짐 게이트: CRASH → 차단, VOLATILE → 강도 반감
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .regime_detector import RegimeState, Regime
from .utils import setup_logger


@dataclass
class TradeSignal:
    """매매 시그널"""
    pair_name: str
    symbol: str
    direction: str                 # "long" or "exit"
    strength: float                # 0.0 ~ 1.0
    us_return: float               # US 전일 수익률
    lead_symbol: str               # US 리드 심볼
    reason: str
    strategy_type: str = "gap_trading"


class GapSignalGenerator:
    """US 종가 모멘텀 → KR 시가 갭 트레이딩 시그널 생성기"""

    def __init__(self, config: dict):
        self.logger = setup_logger("GapSignal")

        gap_cfg = config.get("strategy", {}).get("gap_trading", {})
        self.us_return_threshold = gap_cfg.get("us_return_threshold", 0.005)
        self.us_return_strong = gap_cfg.get("us_return_strong", 0.015)
        self.short_enabled = gap_cfg.get("short_enabled", False)
        self.pair_configs = gap_cfg.get("pairs", [])
        self.max_signals_per_cycle = 6

    def generate(
        self,
        us_returns: dict[str, float],
        regime_state: RegimeState,
        current_positions: set[str],
    ) -> list[TradeSignal]:
        """
        시그널 생성 메인 파이프라인.

        Args:
            us_returns: {US심볼: 전일수익률} (e.g. {"JPM": 0.012, "AAPL": -0.005})
            regime_state: 현재 레짐 상태
            current_positions: 이미 보유 중인 KR 종목 코드 set
        """
        signals = []

        # CRASH → 퇴출 시그널만
        if regime_state.regime == Regime.CRASH:
            return self._generate_exit_signals(regime_state, current_positions)

        # 레짐 배수
        regime_mult = regime_state.position_multiplier  # TREND=1.0, VOLATILE=0.5

        for pair in self.pair_configs:
            lead_sym = pair["lead"]
            lag_symbols = pair["lag"]
            pair_name = pair.get("name", f"{lead_sym}→KR")

            us_ret = us_returns.get(lead_sym)
            if us_ret is None:
                continue

            # 매수 시그널: US return > threshold
            if us_ret >= self.us_return_threshold:
                strength = self._calc_strength(us_ret)
                adjusted = strength * regime_mult

                if adjusted < 0.1:
                    continue

                for sym in lag_symbols:
                    if sym in current_positions:
                        continue
                    signals.append(TradeSignal(
                        pair_name=pair_name,
                        symbol=sym,
                        direction="long",
                        strength=adjusted,
                        us_return=us_ret,
                        lead_symbol=lead_sym,
                        reason=(
                            f"[{pair_name}] {lead_sym} {us_ret:+.2%} > "
                            f"{self.us_return_threshold:.1%}"
                        ),
                    ))

        # 강도순 정렬
        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals[:self.max_signals_per_cycle]

    def _calc_strength(self, us_return: float) -> float:
        """US 수익률 → 시그널 강도 (0~1)"""
        if us_return <= self.us_return_threshold:
            return 0.0

        # 선형 보간: threshold → 0.5, strong → 1.0
        t = self.us_return_threshold
        s = self.us_return_strong
        if us_return >= s:
            return 1.0

        ratio = (us_return - t) / (s - t)
        return 0.5 + 0.5 * ratio

    def _generate_exit_signals(
        self,
        regime_state: RegimeState,
        current_positions: set[str],
    ) -> list[TradeSignal]:
        """CRASH 레짐 퇴출 시그널"""
        signals = []
        if regime_state.regime == Regime.CRASH:
            for symbol in current_positions:
                signals.append(TradeSignal(
                    pair_name="",
                    symbol=symbol,
                    direction="exit",
                    strength=1.0,
                    us_return=0.0,
                    lead_symbol="",
                    reason="CRASH 레짐 전면 매도",
                    strategy_type="crash_exit",
                ))
        return signals
