"""
risk_manager.py - ATR 트레일링 스탑 + 시간 기반 청산 리스크 관리

주요 기능:
  - ATR 기반 트레일링 스탑 (레짐별 배수 조정)
  - 시간 기반 청산 (리드-래그 소멸 감지)
  - 동적 포지션 사이징 (Kelly fraction × 레짐 × 시그널 강도)
  - 일일 거래 한도 및 연속 손실 쿨다운
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .utils import setup_logger, format_krw


@dataclass
class ManagedPosition:
    """리스크 추적이 붙은 관리 포지션"""
    symbol: str
    qty: int
    avg_price: float
    current_price: float
    entry_time: datetime
    entry_regime: str
    pair_name: str

    atr_at_entry: float
    highest_price_since_entry: float
    trailing_stop_level: float
    atr_stop_multiplier: float

    days_held: int = 0
    max_holding_days: int = 3

    @property
    def profit_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price

    @property
    def profit_amount(self) -> float:
        return (self.current_price - self.avg_price) * self.qty


@dataclass
class OrderRequest:
    symbol: str
    side: str
    qty: int
    price: float
    regime: str
    strategy: str
    reason: str
    signal_strength: float = 0.5
    pair_name: str = ""


@dataclass
class OrderDecision:
    approved: bool
    order: Optional[OrderRequest]
    adjusted_qty: int = 0
    rejection_reason: str = ""


class RiskManager:
    """ATR 트레일링 스탑 + 시간 기반 청산 리스크 관리자"""

    def __init__(self, config: dict):
        self.config = config.get("risk", {})
        self.logger = setup_logger("RiskManager")

        self.max_position_pct = self.config.get("max_position_pct", 0.12)
        self.max_exposure = self.config.get("max_total_exposure", 0.80)
        self.atr_period = self.config.get("atr_period", 14)
        self.default_atr_stop_mult = self.config.get("atr_stop_multiplier", 2.0)
        self.max_daily_trades = self.config.get("max_daily_trades", 10)
        self.cooldown_minutes = self.config.get("cooldown_after_loss", 30)
        self.max_consecutive_losses = self.config.get("max_consecutive_losses", 3)
        self.kelly_fraction = self.config.get("kelly_fraction", 0.25)
        self.min_position_krw = self.config.get("min_position_size_krw", 100_000)

        self._managed_positions: dict[str, ManagedPosition] = {}
        self._consecutive_losses = 0
        self._last_loss_time: Optional[datetime] = None
        self._daily_pnl: float = 0.0

    # ──────────────────────────────────────────────
    # 포지션 사이징
    # ──────────────────────────────────────────────

    def calculate_position_size(
        self,
        price: float,
        cash: float,
        total_value: float,
        signal_strength: float,
        regime_multiplier: float,
        atr: float,
        atr_stop_multiplier: float,
    ) -> int:
        """시그널 강도 × 레짐 × ATR 기반 동적 포지션 사이징"""
        if price <= 0 or total_value <= 0 or regime_multiplier <= 0:
            return 0

        base = total_value * self.max_position_pct * self.kelly_fraction
        regime_adjusted = base * regime_multiplier
        strength_clamped = max(0.3, min(signal_strength, 1.0))
        signal_adjusted = regime_adjusted * strength_clamped

        # ATR 기반 수량: 리스크 예산 / 주당 리스크
        risk_per_share = atr * atr_stop_multiplier
        if risk_per_share > 0:
            atr_qty = signal_adjusted / risk_per_share
        else:
            atr_qty = signal_adjusted / price

        # 현금 제약
        cash_qty = (cash * 0.9) / price

        qty = int(min(atr_qty, cash_qty))

        # 최소 금액 체크
        if qty * price < self.min_position_krw:
            return 0

        return max(0, qty)

    # ──────────────────────────────────────────────
    # 포지션 등록
    # ──────────────────────────────────────────────

    def register_position(
        self,
        symbol: str,
        qty: int,
        price: float,
        regime_name: str,
        pair_name: str,
        atr: float,
        atr_stop_multiplier: float,
        max_holding_days: int = 3,
    ) -> ManagedPosition:
        """신규 포지션 등록 (트레일링 스탑 초기화)"""
        stop_level = price - (atr * atr_stop_multiplier)

        pos = ManagedPosition(
            symbol=symbol,
            qty=qty,
            avg_price=price,
            current_price=price,
            entry_time=datetime.now(),
            entry_regime=regime_name,
            pair_name=pair_name,
            atr_at_entry=atr,
            highest_price_since_entry=price,
            trailing_stop_level=stop_level,
            atr_stop_multiplier=atr_stop_multiplier,
            max_holding_days=max_holding_days,
        )

        self._managed_positions[symbol] = pos
        self.logger.info(
            f"포지션 등록: {symbol} {qty}주 @ {price:,.0f} | "
            f"ATR스탑: {stop_level:,.0f} | 최대보유: {max_holding_days}일"
        )
        return pos

    # ──────────────────────────────────────────────
    # 포지션 업데이트 & 스탑 체크
    # ──────────────────────────────────────────────

    def update_positions(
        self,
        current_prices: dict[str, float],
    ) -> list[tuple[ManagedPosition, str]]:
        """모든 관리 포지션 업데이트 및 청산 조건 체크"""
        to_close = []

        for symbol in list(self._managed_positions.keys()):
            pos = self._managed_positions[symbol]

            if symbol in current_prices:
                pos.current_price = current_prices[symbol]

            # 트레일링 스탑 업데이트
            if pos.current_price > pos.highest_price_since_entry:
                pos.highest_price_since_entry = pos.current_price
                new_stop = pos.highest_price_since_entry - (
                    pos.atr_at_entry * pos.atr_stop_multiplier
                )
                pos.trailing_stop_level = max(pos.trailing_stop_level, new_stop)

            # 트레일링 스탑 히트
            if pos.current_price <= pos.trailing_stop_level:
                reason = (
                    f"ATR 트레일링 스탑 ({pos.current_price:,.0f} <= "
                    f"{pos.trailing_stop_level:,.0f})"
                )
                to_close.append((pos, reason))
                self.logger.warning(
                    f"스탑 발동: {symbol} | {pos.profit_pct:.2%} | {reason}"
                )
                continue

            # 시간 기반 청산
            pos.days_held += 1
            if pos.days_held >= pos.max_holding_days:
                reason = f"보유기한 초과 ({pos.days_held}일 >= {pos.max_holding_days}일)"
                to_close.append((pos, reason))
                self.logger.info(f"시간 청산: {symbol} | {pos.profit_pct:.2%} | {reason}")

        return to_close

    def remove_position(self, symbol: str):
        """청산된 포지션 제거"""
        self._managed_positions.pop(symbol, None)

    def force_liquidate_all(self) -> list[tuple[ManagedPosition, str]]:
        """CRASH 레짐: 전체 포지션 강제 청산"""
        to_close = []
        for symbol, pos in self._managed_positions.items():
            to_close.append((pos, "CRASH 레짐 강제 청산"))
        return to_close

    def get_managed_positions(self) -> dict[str, ManagedPosition]:
        return self._managed_positions.copy()

    # ──────────────────────────────────────────────
    # 주문 검증
    # ──────────────────────────────────────────────

    def validate_order(
        self,
        request: OrderRequest,
        cash: float,
        total_value: float,
        positions: list,
        daily_trade_count: int,
    ) -> OrderDecision:
        """주문 리스크 검증"""
        if daily_trade_count >= self.max_daily_trades:
            return OrderDecision(
                approved=False, order=request,
                rejection_reason=f"일일 거래 한도 ({self.max_daily_trades}회)",
            )

        if self._is_in_cooldown():
            remaining = self._cooldown_remaining()
            return OrderDecision(
                approved=False, order=request,
                rejection_reason=f"쿨다운 중 ({remaining}분 남음)",
            )

        if request.side == "sell":
            return OrderDecision(
                approved=True, order=request, adjusted_qty=request.qty,
            )

        # 매수 검증
        order_amount = request.price * request.qty

        if order_amount > cash * 0.95:
            adjusted_qty = int((cash * 0.95) / request.price)
            if adjusted_qty <= 0:
                return OrderDecision(
                    approved=False, order=request,
                    rejection_reason=f"현금 부족 ({format_krw(cash)})",
                )
            request.qty = adjusted_qty
            order_amount = request.price * adjusted_qty

        # 종목 비중 한도
        existing_amount = 0
        for pos in positions:
            if pos.symbol == request.symbol:
                existing_amount = pos.current_price * pos.qty
                break

        new_total = existing_amount + order_amount
        max_allowed = total_value * self.max_position_pct

        if new_total > max_allowed:
            allowed = max(0, max_allowed - existing_amount)
            adjusted_qty = int(allowed / request.price)
            if adjusted_qty <= 0:
                return OrderDecision(
                    approved=False, order=request,
                    rejection_reason=f"종목 비중 한도 ({self.max_position_pct:.0%})",
                )
            request.qty = adjusted_qty

        # 전체 익스포저 한도
        total_invested = sum(p.current_price * p.qty for p in positions)
        new_exposure = (total_invested + order_amount) / total_value if total_value > 0 else 0
        if new_exposure > self.max_exposure:
            return OrderDecision(
                approved=False, order=request,
                rejection_reason=f"전체 익스포저 한도 ({new_exposure:.1%} > {self.max_exposure:.1%})",
            )

        return OrderDecision(
            approved=True, order=request, adjusted_qty=request.qty,
        )

    # ──────────────────────────────────────────────
    # 연속 손실 / 쿨다운
    # ──────────────────────────────────────────────

    def record_trade_result(self, profit: float):
        self._daily_pnl += profit
        if profit < 0:
            self._consecutive_losses += 1
            self._last_loss_time = datetime.now()
            if self._consecutive_losses >= self.max_consecutive_losses:
                self.logger.warning(
                    f"연속 {self._consecutive_losses}회 손실 → {self.cooldown_minutes}분 쿨다운"
                )
        else:
            self._consecutive_losses = 0

    def _is_in_cooldown(self) -> bool:
        if self._consecutive_losses < self.max_consecutive_losses:
            return False
        if self._last_loss_time is None:
            return False
        elapsed = (datetime.now() - self._last_loss_time).total_seconds() / 60
        return elapsed < self.cooldown_minutes

    def _cooldown_remaining(self) -> int:
        if self._last_loss_time is None:
            return 0
        elapsed = (datetime.now() - self._last_loss_time).total_seconds() / 60
        return max(0, int(self.cooldown_minutes - elapsed))

    def reset_daily(self):
        self._daily_pnl = 0.0
        self.logger.info("일일 리스크 카운터 리셋")

    def get_status(self) -> dict:
        return {
            "managed_positions": len(self._managed_positions),
            "consecutive_losses": self._consecutive_losses,
            "in_cooldown": self._is_in_cooldown(),
            "cooldown_remaining_min": self._cooldown_remaining(),
            "daily_pnl": self._daily_pnl,
        }

    def get_positions_report(self) -> str:
        lines = []
        for symbol, pos in self._managed_positions.items():
            lines.append(
                f"  {symbol} | {pos.qty}주 | "
                f"진입: {pos.avg_price:,.0f} | 현재: {pos.current_price:,.0f} | "
                f"수익: {pos.profit_pct:+.2%} | "
                f"스탑: {pos.trailing_stop_level:,.0f} | "
                f"보유: {pos.days_held}/{pos.max_holding_days}일"
            )
        return "\n".join(lines) if lines else "  (없음)"
