"""
kis_trader.py - 한국투자증권 KIS API 래퍼
python-kis (pykis) 라이브러리를 활용한 모의투자/실전투자 주문 모듈
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
from pykis import PyKis, KisAuth

from .utils import setup_logger


@dataclass
class Position:
    """보유 포지션 정보"""
    symbol: str
    market: str          # "KRX" or "NYSE" / "NASDAQ"
    qty: int
    avg_price: float
    current_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def profit_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price

    @property
    def profit_amount(self) -> float:
        return (self.current_price - self.avg_price) * self.qty


@dataclass
class TradeRecord:
    """거래 기록"""
    timestamp: datetime
    symbol: str
    market: str
    side: str             # "buy" or "sell"
    qty: int
    price: float
    regime: str           # 거래 시점의 레짐
    strategy: str         # 사용한 전략
    reason: str           # 거래 사유


class KisTrader:
    """
    KIS API 트레이딩 래퍼
    - 모의투자/실전투자 자동 전환
    - 국내/해외 주식 통합 인터페이스
    - 호출 제한 (초당 20건) 자동 관리
    """

    def __init__(self, config: dict):
        self.config = config["kis"]
        self.logger = setup_logger("KisTrader")
        self.trade_history: list[TradeRecord] = []
        self.daily_trade_count = 0
        self._last_call_time = 0.0
        self._call_interval = 0.05  # 초당 20건 → 50ms 간격

        self._init_client()

    def _init_client(self):
        """PyKis 클라이언트 초기화"""
        cfg = self.config

        if cfg.get("is_virtual", True):
            # 모의투자
            self.kis = PyKis(
                id=cfg["id"],
                account=cfg["account"],
                appkey=cfg["appkey"],
                secretkey=cfg["secretkey"],
                virtual_id=cfg.get("virtual_id", cfg["id"]),
                virtual_appkey=cfg["virtual_appkey"],
                virtual_secretkey=cfg["virtual_secretkey"],
                keep_token=True,
            )
            self.logger.info("모의투자 모드로 연결됨")
        else:
            # 실전투자
            self.kis = PyKis(
                id=cfg["id"],
                account=cfg["account"],
                appkey=cfg["appkey"],
                secretkey=cfg["secretkey"],
                keep_token=True,
            )
            self.logger.info("실전투자 모드로 연결됨")

    def _rate_limit(self):
        """API 호출 레이트 리밋 관리"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._call_interval:
            time.sleep(self._call_interval - elapsed)
        self._last_call_time = time.time()

    def get_order_condition(self) -> Optional[str]:
        """
        현재 시각에 맞는 주문 condition 자동 판별.
        Returns:
            None       → 정규장 (09:00~15:30)
            'before'   → 장전시간외 (08:30~08:40, 전일종가)
            'after'    → 장후시간외 (15:40~16:00, 당일종가)
            'extended' → 시간외단일가 (16:00~18:00, 지정가)
            False      → 주문 불가 시간
        """
        now = datetime.now()
        h, m = now.hour, now.minute
        t = h * 60 + m  # 분 단위

        if now.weekday() >= 5:
            return False

        if 510 <= t < 520:      # 08:30~08:40
            return "before"
        elif 540 <= t < 930:    # 09:00~15:30
            return None
        elif 940 <= t < 960:    # 15:40~16:00
            return "after"
        elif 960 <= t < 1080:   # 16:00~18:00
            return "extended"
        else:
            return False

    def is_orderable(self) -> bool:
        """현재 주문 가능한 시간인지"""
        return self.get_order_condition() is not False

    # ──────────────────────────────────────────────
    # 시세 조회
    # ──────────────────────────────────────────────

    def get_price(self, symbol: str) -> dict:
        """
        종목 현재가 조회 (국내/해외 자동 판별)
        Returns: {"price": float, "change_pct": float, "volume": int, ...}
        """
        self._rate_limit()
        try:
            stock = self.kis.stock(symbol)
            quote = stock.quote()
            return {
                "symbol": symbol,
                "price": float(quote.close),
                "open": float(quote.open),
                "high": float(quote.high),
                "low": float(quote.low),
                "change_pct": float(quote.rate) if hasattr(quote, "rate") else 0.0,
                "volume": int(quote.volume) if hasattr(quote, "volume") else 0,
            }
        except Exception as e:
            self.logger.error(f"시세 조회 실패 [{symbol}]: {e}")
            return {}

    def get_daily_prices(
        self, symbol: str, days: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        일봉 데이터 조회
        Returns: DataFrame with columns [open, high, low, close, volume]
        """
        self._rate_limit()
        try:
            stock = self.kis.stock(symbol)
            chart = stock.chart(period="D")

            records = []
            for candle in chart:
                records.append({
                    "date": candle.date if hasattr(candle, "date") else candle.time,
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": int(candle.volume) if hasattr(candle, "volume") else 0,
                })

            df = pd.DataFrame(records)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").tail(days).reset_index(drop=True)
            return df

        except Exception as e:
            self.logger.error(f"일봉 조회 실패 [{symbol}]: {e}")
            return None

    # ──────────────────────────────────────────────
    # 주문 실행
    # ──────────────────────────────────────────────

    # 시간대별 주문 타입 라벨
    _CONDITION_LABELS = {
        None: "정규장",
        "before": "장전시간외",
        "after": "장후시간외",
        "extended": "시간외단일가",
    }

    def buy(
        self,
        symbol: str,
        qty: int,
        price: Optional[float] = None,
        condition: Optional[str] = "auto",
        regime: str = "",
        strategy: str = "",
        reason: str = "",
    ) -> bool:
        """
        매수 주문 (시간대 자동 판별)
        price=None → 시장가, price=값 → 지정가
        condition='auto' → 현재 시각에 맞게 자동 결정
        condition=None/'before'/'after'/'extended' → 직접 지정
        """
        self._rate_limit()

        # 주문 condition 결정
        if condition == "auto":
            condition = self.get_order_condition()
            if condition is False:
                self.logger.warning(f"매수 스킵 [{symbol}]: 주문 불가 시간")
                return False

        # 시간외단일가는 반드시 지정가 필요
        if condition == "extended" and price is None:
            price_data = self.get_price(symbol)
            price = price_data.get("price", 0) if price_data else 0
            if price <= 0:
                self.logger.error(f"매수 실패 [{symbol}]: 시간외단일가 가격 조회 실패")
                return False

        cond_label = self._CONDITION_LABELS.get(condition, condition)

        try:
            stock = self.kis.stock(symbol)

            if condition:
                # 시간외 주문
                if price:
                    order = stock.buy(price=price, qty=qty, condition=condition)
                else:
                    order = stock.buy(qty=qty, condition=condition)
            else:
                # 정규장 주문
                if price:
                    order = stock.buy(price=price, qty=qty)
                else:
                    order = stock.buy(qty=qty)

            executed_price = price or self.get_price(symbol).get("price", 0)

            record = TradeRecord(
                timestamp=datetime.now(),
                symbol=symbol,
                market=self._detect_market(symbol),
                side="buy",
                qty=qty,
                price=executed_price,
                regime=regime,
                strategy=strategy,
                reason=reason,
            )
            self.trade_history.append(record)
            self.daily_trade_count += 1

            self.logger.info(
                f"매수 | {symbol} | {qty}주 | "
                f"{'시장가' if not price else f'{price:,.0f}'} | "
                f"{cond_label} | [{regime}/{strategy}] {reason}"
            )
            return True

        except Exception as e:
            self.logger.error(f"매수 실패 [{symbol}] ({cond_label}): {e}")
            return False

    def sell(
        self,
        symbol: str,
        qty: Optional[int] = None,
        price: Optional[float] = None,
        condition: Optional[str] = "auto",
        regime: str = "",
        strategy: str = "",
        reason: str = "",
    ) -> bool:
        """
        매도 주문 (시간대 자동 판별)
        qty=None → 전량 매도
        condition='auto' → 현재 시각에 맞게 자동 결정
        """
        self._rate_limit()

        # 주문 condition 결정
        if condition == "auto":
            condition = self.get_order_condition()
            if condition is False:
                self.logger.warning(f"매도 스킵 [{symbol}]: 주문 불가 시간")
                return False

        # 시간외단일가는 반드시 지정가 필요
        if condition == "extended" and price is None:
            price_data = self.get_price(symbol)
            price = price_data.get("price", 0) if price_data else 0
            if price <= 0:
                self.logger.error(f"매도 실패 [{symbol}]: 시간외단일가 가격 조회 실패")
                return False

        cond_label = self._CONDITION_LABELS.get(condition, condition)

        try:
            stock = self.kis.stock(symbol)

            if condition:
                # 시간외 주문
                if price and qty:
                    order = stock.sell(price=price, qty=qty, condition=condition)
                elif qty:
                    order = stock.sell(qty=qty, condition=condition)
                else:
                    order = stock.sell(condition=condition)
            else:
                # 정규장 주문
                if price and qty:
                    order = stock.sell(price=price, qty=qty)
                elif qty:
                    order = stock.sell(qty=qty)
                else:
                    order = stock.sell()

            executed_price = price or self.get_price(symbol).get("price", 0)

            record = TradeRecord(
                timestamp=datetime.now(),
                symbol=symbol,
                market=self._detect_market(symbol),
                side="sell",
                qty=qty or 0,
                price=executed_price,
                regime=regime,
                strategy=strategy,
                reason=reason,
            )
            self.trade_history.append(record)
            self.daily_trade_count += 1

            self.logger.info(
                f"매도 | {symbol} | {qty or '전량'}주 | "
                f"{'시장가' if not price else f'{price:,.0f}'} | "
                f"{cond_label} | [{regime}/{strategy}] {reason}"
            )
            return True

        except Exception as e:
            self.logger.error(f"매도 실패 [{symbol}] ({cond_label}): {e}")
            return False

    # ──────────────────────────────────────────────
    # 계좌 조회
    # ──────────────────────────────────────────────

    def get_balance(self) -> dict:
        """
        계좌 잔고 조회
        Returns: {"cash": float, "total": float, "positions": [Position, ...]}
        """
        self._rate_limit()
        try:
            account = self.kis.account()
            balance = account.balance()

            positions = []
            for stock in balance.stocks:
                positions.append(Position(
                    symbol=stock.symbol,
                    market=str(stock.market),
                    qty=int(stock.qty),
                    avg_price=float(stock.price),
                    current_price=float(stock.amount / stock.qty) if stock.qty > 0 else 0,
                ))

            # 예수금 조회
            total_value = float(balance.purchase_amount + balance.profit)
            cash = 0.0
            for deposit in balance.deposits.values():
                cash += float(deposit.amount)

            return {
                "cash": cash,
                "total": total_value,
                "positions": positions,
                "profit": float(balance.profit),
                "profit_pct": float(balance.profit_rate) * 100,
            }

        except Exception as e:
            self.logger.error(f"잔고 조회 실패: {e}")
            return {"cash": 0, "total": 0, "positions": [], "profit": 0, "profit_pct": 0}

    def get_pending_orders(self) -> list:
        """미체결 주문 조회"""
        self._rate_limit()
        try:
            account = self.kis.account()
            return list(account.pending_orders())
        except Exception as e:
            self.logger.error(f"미체결 조회 실패: {e}")
            return []

    def cancel_all_pending(self):
        """미체결 주문 전체 취소"""
        orders = self.get_pending_orders()
        for order in orders:
            try:
                order.cancel()
                self.logger.info(f"주문 취소: {order}")
            except Exception as e:
                self.logger.error(f"주문 취소 실패: {e}")

    # ──────────────────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────────────────

    @staticmethod
    def _detect_market(symbol: str) -> str:
        """종목코드로 시장 판별 (숫자 6자리 = KRX, 영문 = US)"""
        return "KRX" if symbol.isdigit() and len(symbol) == 6 else "US"

    def reset_daily_count(self):
        """일일 거래 카운트 리셋"""
        self.daily_trade_count = 0

    def get_trade_summary(self) -> pd.DataFrame:
        """거래 내역 요약"""
        if not self.trade_history:
            return pd.DataFrame()
        records = [
            {
                "time": t.timestamp,
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.qty,
                "price": t.price,
                "regime": t.regime,
                "strategy": t.strategy,
                "reason": t.reason,
            }
            for t in self.trade_history
        ]
        return pd.DataFrame(records)
