"""
strategy.py - Regime-Gated Gap Trading 통합 전략 엔진

US 전일 종가 모멘텀 → KR 시가 갭 트레이딩.

흐름:
  1. 레짐 감지 (변동성 + HMM) → TREND / VOLATILE / CRASH
  2. US 전일 수익률 계산
  3. 갭 시그널 생성 (레짐 게이트 적용)
  4. 리스크 관리 (청산 모드별 처리)
  5. 주문 실행
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from .regime_detector import CompositeRegimeDetector, RegimeState, Regime
from .signal_generator import GapSignalGenerator, TradeSignal
from .risk_manager import RiskManager, OrderRequest
from .kis_trader import KisTrader
from .utils import (
    setup_logger, calc_atr, is_kr_market_open, send_telegram, format_krw,
)


@dataclass
class StrategyDecision:
    """매매 의사결정 결과"""
    action: str
    symbol: str
    qty: int
    price: float
    regime: str
    strategy: str
    reason: str
    signal_strength: float
    confidence: float
    pair_name: str = ""


class RegimeGatedStrategy:
    """Regime-Gated Gap Trading 통합 전략"""

    def __init__(self, config: dict, trader: KisTrader):
        self.config = config
        self.trader = trader
        self.logger = setup_logger("Strategy")

        strategy_cfg = config.get("strategy", {})

        self.regime_detector = CompositeRegimeDetector(strategy_cfg.get("regime", {}))
        self.signal_generator = GapSignalGenerator(config)
        self.risk_manager = RiskManager(config)

        gap_cfg = strategy_cfg.get("gap_trading", {})
        self._pair_configs = gap_cfg.get("pairs", [])
        self._exit_mode = gap_cfg.get("exit_mode", "next_close")
        self._max_holding_days = gap_cfg.get("max_holding_days", 2)

        self._benchmark_symbol = "005930"
        self._current_regime_state: Optional[RegimeState] = None

        noti = config.get("notification", {})
        self.tg_token = noti.get("telegram_bot_token", "")
        self.tg_chat = noti.get("telegram_chat_id", "")

    # ──────────────────────────────────────────────
    # 메인 사이클
    # ──────────────────────────────────────────────

    def run_cycle(self) -> list[StrategyDecision]:
        """전략 사이클 1회 실행"""
        decisions = []

        # 1. 레짐 업데이트
        self._update_regime()
        if self._current_regime_state is None:
            self.logger.warning("레짐 판별 불가 → 스킵")
            return decisions

        regime = self._current_regime_state.regime
        self.logger.info(
            f"레짐: {regime.value} | "
            f"확신도: {self._current_regime_state.confidence:.1%} | "
            f"변동성: {self._current_regime_state.vol_long:.1%} | "
            f"추세: {self._current_regime_state.trend_direction}"
        )

        # 2. CRASH → 전량 청산
        if regime == Regime.CRASH:
            decisions.extend(self._handle_crash())
            self._execute_decisions(decisions)
            return decisions

        # 3. 기존 포지션 청산 체크 (exit_mode에 따라)
        current_prices = self._get_current_prices()
        stop_exits = self.risk_manager.update_positions(current_prices)
        for pos, reason in stop_exits:
            decisions.append(StrategyDecision(
                action="sell", symbol=pos.symbol, qty=pos.qty,
                price=pos.current_price, regime=regime.value,
                strategy="trailing_stop" if "ATR" in reason else "time_exit",
                reason=reason, signal_strength=1.0, confidence=1.0,
                pair_name=pos.pair_name,
            ))
            self.risk_manager.remove_position(pos.symbol)

        # 4. US 전일 수익률 조회
        us_returns = self._get_us_returns()
        if us_returns:
            self.logger.info(f"US 수익률: {', '.join(f'{k}:{v:+.2%}' for k, v in us_returns.items())}")
        else:
            self.logger.info("US 수익률 데이터 없음")

        # 5. 갭 시그널 생성
        held_symbols = set(self.risk_manager.get_managed_positions().keys())
        balance = self.trader.get_balance()
        for pos in balance.get("positions", []):
            held_symbols.add(pos.symbol)

        signals = self.signal_generator.generate(
            us_returns=us_returns,
            regime_state=self._current_regime_state,
            current_positions=held_symbols,
        )

        # 6. 시그널 → 주문 결정
        for signal in signals:
            if signal.direction == "exit":
                qty = self._get_held_qty(signal.symbol)
                if qty > 0:
                    decisions.append(StrategyDecision(
                        action="sell", symbol=signal.symbol, qty=qty,
                        price=current_prices.get(signal.symbol, 0),
                        regime=regime.value,
                        strategy=signal.strategy_type, reason=signal.reason,
                        signal_strength=signal.strength,
                        confidence=self._current_regime_state.confidence,
                        pair_name=signal.pair_name,
                    ))
                    self.risk_manager.remove_position(signal.symbol)

            elif signal.direction == "long":
                price = current_prices.get(signal.symbol, 0)
                if price <= 0:
                    continue

                atr = self._get_atr(signal.symbol)
                atr_mult = self._current_regime_state.atr_stop_multiplier

                qty = self.risk_manager.calculate_position_size(
                    price=price,
                    cash=balance.get("cash", 0),
                    total_value=balance.get("total", 0),
                    signal_strength=signal.strength,
                    regime_multiplier=self._current_regime_state.position_multiplier,
                    atr=atr,
                    atr_stop_multiplier=atr_mult,
                )

                if qty > 0:
                    decisions.append(StrategyDecision(
                        action="buy", symbol=signal.symbol, qty=qty,
                        price=price, regime=regime.value,
                        strategy=signal.strategy_type, reason=signal.reason,
                        signal_strength=signal.strength,
                        confidence=self._current_regime_state.confidence,
                        pair_name=signal.pair_name,
                    ))

        # 7. 실행
        self._execute_decisions(decisions)
        return decisions

    # ──────────────────────────────────────────────
    # CRASH 처리
    # ──────────────────────────────────────────────

    def _handle_crash(self) -> list[StrategyDecision]:
        """CRASH 레짐 전량 청산"""
        decisions = []

        liquidations = self.risk_manager.force_liquidate_all()
        for pos, reason in liquidations:
            decisions.append(StrategyDecision(
                action="sell", symbol=pos.symbol, qty=pos.qty,
                price=pos.current_price, regime="CRASH",
                strategy="crash_exit", reason=reason,
                signal_strength=1.0, confidence=1.0,
                pair_name=pos.pair_name,
            ))
            self.risk_manager.remove_position(pos.symbol)

        balance = self.trader.get_balance()
        managed = set(self.risk_manager.get_managed_positions().keys())
        for pos in balance.get("positions", []):
            if pos.symbol not in managed and pos.qty > 0:
                decisions.append(StrategyDecision(
                    action="sell", symbol=pos.symbol, qty=pos.qty,
                    price=pos.current_price, regime="CRASH",
                    strategy="crash_exit", reason="CRASH 레짐 전면 매도",
                    signal_strength=1.0, confidence=1.0,
                ))

        if decisions:
            self.logger.warning(f"CRASH 레짐: {len(decisions)}개 포지션 강제 청산")
            send_telegram(
                self.tg_token, self.tg_chat,
                f"CRASH 레짐 감지! {len(decisions)}개 포지션 강제 청산 중"
            )

        return decisions

    # ──────────────────────────────────────────────
    # 데이터 수집
    # ──────────────────────────────────────────────

    def _get_us_returns(self) -> dict[str, float]:
        """US 리드 심볼들의 전일 수익률 조회"""
        us_returns = {}
        seen = set()
        for pair in self._pair_configs:
            symbol = pair["lead"]
            if symbol in seen:
                continue
            seen.add(symbol)

            df = self.trader.get_daily_prices(symbol, days=5)
            if df is not None and len(df) >= 2:
                close_today = df["close"].iloc[-1]
                close_prev = df["close"].iloc[-2]
                if close_prev > 0:
                    us_returns[symbol] = (close_today / close_prev) - 1

        return us_returns

    def _get_current_prices(self) -> dict[str, float]:
        """모든 래그 종목 현재가 조회"""
        prices = {}
        for pair in self._pair_configs:
            for symbol in pair["lag"]:
                if symbol not in prices:
                    data = self.trader.get_price(symbol)
                    if data:
                        prices[symbol] = data.get("price", 0)
        return prices

    def _get_atr(self, symbol: str) -> float:
        """종목의 ATR(14) 계산"""
        df = self.trader.get_daily_prices(symbol, days=30)
        if df is not None and len(df) >= 15:
            atr_series = calc_atr(df["high"], df["low"], df["close"], 14)
            val = atr_series.iloc[-1]
            if not pd.isna(val):
                return float(val)
        return 0.0

    def _get_held_qty(self, symbol: str) -> int:
        """보유 수량 조회"""
        managed = self.risk_manager.get_managed_positions()
        if symbol in managed:
            return managed[symbol].qty
        balance = self.trader.get_balance()
        for pos in balance.get("positions", []):
            if pos.symbol == symbol:
                return pos.qty
        return 0

    # ──────────────────────────────────────────────
    # 레짐 관리
    # ──────────────────────────────────────────────

    def _update_regime(self):
        """벤치마크 기반 레짐 업데이트"""
        if self.regime_detector.needs_retrain():
            df = self.trader.get_daily_prices(self._benchmark_symbol, days=120)
            if df is not None and len(df) > 30:
                self.regime_detector.fit_hmm(df["close"])

        df = self.trader.get_daily_prices(self._benchmark_symbol, days=120)
        if df is not None and len(df) > 20:
            self._current_regime_state = self.regime_detector.detect(df["close"])

    # ──────────────────────────────────────────────
    # 주문 실행
    # ──────────────────────────────────────────────

    def _execute_decisions(self, decisions: list[StrategyDecision]):
        """매매 결정 실행"""
        if not decisions:
            return

        balance = self.trader.get_balance()

        for d in decisions:
            order_req = OrderRequest(
                symbol=d.symbol, side=d.action, qty=d.qty, price=d.price,
                regime=d.regime, strategy=d.strategy, reason=d.reason,
                signal_strength=d.signal_strength, pair_name=d.pair_name,
            )

            verdict = self.risk_manager.validate_order(
                request=order_req,
                cash=balance.get("cash", 0),
                total_value=balance.get("total", 0),
                positions=balance.get("positions", []),
                daily_trade_count=self.trader.daily_trade_count,
            )

            if not verdict.approved:
                self.logger.info(
                    f"주문 거부: {d.action} {d.symbol} | {verdict.rejection_reason}"
                )
                continue

            success = False
            if d.action == "buy":
                final_qty = verdict.adjusted_qty or d.qty
                success = self.trader.buy(
                    symbol=d.symbol, qty=final_qty,
                    regime=d.regime, strategy=d.strategy, reason=d.reason,
                )
                if success:
                    atr_mult = self._current_regime_state.atr_stop_multiplier if self._current_regime_state else 2.0
                    atr_val = d.price * 0.02
                    managed = self.risk_manager.get_managed_positions()
                    if d.symbol not in managed:
                        self.risk_manager.register_position(
                            symbol=d.symbol, qty=final_qty, price=d.price,
                            regime_name=d.regime, pair_name=d.pair_name,
                            atr=atr_val, atr_stop_multiplier=atr_mult,
                            max_holding_days=self._max_holding_days,
                        )

            elif d.action == "sell":
                success = self.trader.sell(
                    symbol=d.symbol,
                    qty=d.qty if d.qty > 0 else None,
                    regime=d.regime, strategy=d.strategy, reason=d.reason,
                )
                if success:
                    self.risk_manager.remove_position(d.symbol)
                    managed = self.risk_manager.get_managed_positions()
                    if d.symbol in managed:
                        pos = managed[d.symbol]
                        profit = (d.price - pos.avg_price) * pos.qty
                        self.risk_manager.record_trade_result(profit)

            if success:
                send_telegram(
                    self.tg_token, self.tg_chat,
                    f"{d.action.upper()} {d.symbol} {d.qty}주 | "
                    f"[{d.strategy}] {d.reason}"
                )

    # ──────────────────────────────────────────────
    # 리포트
    # ──────────────────────────────────────────────

    def get_pre_market_report(self) -> str:
        """프리마켓 US 수익률 분석 리포트"""
        lines = [
            f"=== Gap Trading 프리마켓 분석 ({datetime.now().strftime('%H:%M')}) ===",
        ]

        if self._current_regime_state:
            rs = self._current_regime_state
            lines.append(
                f"레짐: {rs.regime.value} | "
                f"변동성: {rs.vol_long:.1%} | 추세: {rs.trend_direction}"
            )

        lines.append("")

        us_returns = self._get_us_returns()
        if us_returns:
            lines.append("US 전일 수익률:")
            threshold = self.signal_generator.us_return_threshold
            for sym, ret in sorted(us_returns.items(), key=lambda x: x[1], reverse=True):
                flag = " <<< 시그널!" if ret >= threshold else ""
                lines.append(f"  {sym:6s} {ret:+.2%}{flag}")
        else:
            lines.append("(US 수익률 데이터 없음)")

        return "\n".join(lines)

    def get_status_report(self) -> str:
        """종합 상태 리포트"""
        lines = ["=" * 50, "Regime-Gated Gap Trading Bot 상태", "=" * 50]

        if self._current_regime_state:
            rs = self._current_regime_state
            lines.append(f"\n레짐: {rs.regime.value}")
            lines.append(f"  확신도: {rs.confidence:.1%}")
            lines.append(f"  단기변동성: {rs.vol_short:.1%}")
            lines.append(f"  장기변동성: {rs.vol_long:.1%}")
            lines.append(f"  VoV: {rs.vol_of_vol:.2f}")
            lines.append(f"  추세: {rs.trend_direction}")
            lines.append(f"  지속: {rs.duration}일")

        lines.append(f"\n전략: Gap Trading ({self._exit_mode})")

        balance = self.trader.get_balance()
        lines.append(f"\n계좌:")
        lines.append(f"  현금: {format_krw(balance.get('cash', 0))}")
        lines.append(f"  총평가: {format_krw(balance.get('total', 0))}")
        lines.append(f"  수익률: {balance.get('profit_pct', 0):+.2f}%")

        lines.append(f"\n관리 포지션:")
        lines.append(self.risk_manager.get_positions_report())

        risk_status = self.risk_manager.get_status()
        lines.append(f"\n리스크:")
        lines.append(f"  관리 포지션: {risk_status['managed_positions']}개")
        lines.append(f"  연속손실: {risk_status['consecutive_losses']}회")
        lines.append(f"  쿨다운: {'활성' if risk_status['in_cooldown'] else '비활성'}")
        lines.append(f"  일일 거래: {self.trader.daily_trade_count}회")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)
