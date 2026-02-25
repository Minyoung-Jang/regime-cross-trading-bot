"""
backtest.py - Gap Trading Backtest Engine

US 전일 종가 모멘텀 → KR 시가 갭 트레이딩 백테스트.

청산 모드:
  - intraday:   KR 시가 매수 → 당일 종가 매도
  - next_close: KR 시가 매수 → 다음날 종가 매도
  - swing:      KR 시가 매수 → ATR 트레일링 스탑 + 최대 N일 보유
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .regime_detector import CompositeRegimeDetector, Regime, RegimeState
from .signal_generator import GapSignalGenerator, TradeSignal
from .utils import setup_logger, calc_atr, format_krw


@dataclass
class BacktestTrade:
    date: datetime
    symbol: str
    side: str
    qty: int
    price: float
    regime: str
    strategy: str
    reason: str
    pair_name: str = ""


@dataclass
class BacktestPosition:
    symbol: str
    qty: int
    avg_price: float
    entry_idx: int
    pair_name: str
    atr_at_entry: float
    highest_price: float
    trailing_stop: float
    atr_stop_mult: float
    max_holding_days: int


@dataclass
class BacktestResult:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    avg_holding_days: float
    profit_factor: float
    trades: list[BacktestTrade]
    equity_curve: pd.Series
    regime_returns: dict
    pair_returns: dict
    regime_distribution: dict

    def summary(self) -> str:
        lines = [
            f"{'=' * 42}",
            f"   Gap Trading 백테스트 결과",
            f"{'=' * 42}",
            f"  총 수익률:        {self.total_return:+.2%}",
            f"  연환산 수익률:    {self.annualized_return:+.2%}",
            f"  최대 낙폭(MDD):   {self.max_drawdown:.2%}",
            f"  샤프 비율:        {self.sharpe_ratio:.2f}",
            f"  승률:             {self.win_rate:.1%}",
            f"  총 거래:          {self.total_trades}회",
            f"  평균 보유:        {self.avg_holding_days:.1f}일",
            f"  이익/손실 비율:   {self.profit_factor:.2f}",
            f"{'-' * 42}",
            f"  레짐 분포:",
        ]
        for k, v in self.regime_distribution.items():
            lines.append(f"    {k}: {v:.1%}")

        lines.append(f"{'-' * 42}")
        lines.append(f"  레짐별 수익률:")
        for k, v in self.regime_returns.items():
            lines.append(f"    {k}: {v:+.2%}")

        if self.pair_returns:
            lines.append(f"{'-' * 42}")
            lines.append(f"  페어별 수익률:")
            for k, v in sorted(self.pair_returns.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"    {k}: {v:+.2%}")

        lines.append(f"{'=' * 42}")
        return "\n".join(lines)


class Backtester:
    """Gap Trading 백테스트 엔진"""

    def __init__(self, config: dict):
        self.config = config
        strategy_cfg = config.get("strategy", {})
        self.logger = setup_logger("Backtest")

        self.regime_detector = CompositeRegimeDetector(strategy_cfg.get("regime", {}))
        self.signal_generator = GapSignalGenerator(config)

        gap_cfg = strategy_cfg.get("gap_trading", {})
        self.exit_mode = gap_cfg.get("exit_mode", "next_close")
        self.max_holding_days = gap_cfg.get("max_holding_days", 2)
        self.pair_configs = gap_cfg.get("pairs", [])

        risk_cfg = config.get("risk", {})
        self.max_position_pct = risk_cfg.get("max_position_pct", 0.12)
        self.kelly_fraction = risk_cfg.get("kelly_fraction", 0.25)
        self.max_exposure = risk_cfg.get("max_total_exposure", 0.80)
        self.default_atr_stop = risk_cfg.get("atr_stop_multiplier", 2.0)

        self._kr_data: dict[str, pd.DataFrame] = {}
        self._us_data: dict[str, pd.DataFrame] = {}

    def load_csv(self, filepaths: dict[str, str]):
        for symbol, path in filepaths.items():
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            self._kr_data[symbol] = df
        self.logger.info(f"KR 데이터 로드: {len(self._kr_data)}종목")

    def load_us_csv(self, filepaths: dict[str, str]):
        for symbol, path in filepaths.items():
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            self._us_data[symbol] = df
        self.logger.info(f"US 데이터 로드: {len(self._us_data)}종목")

    def load_data(self, data: dict[str, pd.DataFrame]):
        for symbol, df in data.items():
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            self._kr_data[symbol] = df
        self.logger.info(f"데이터 로드: {len(data)}종목")

    def run(
        self,
        initial_cash: float = 500_000_000,
        benchmark_symbol: str = "005930",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BacktestResult:
        if not self._kr_data:
            raise ValueError("데이터를 먼저 로드하세요")

        if benchmark_symbol not in self._kr_data:
            benchmark_symbol = list(self._kr_data.keys())[0]

        filter_start = pd.Timestamp(start_date) if start_date else None
        filter_end = pd.Timestamp(end_date) if end_date else None

        self.logger.info(
            f"Gap Trading 백테스트 | 초기자금: {format_krw(initial_cash)} | "
            f"청산모드: {self.exit_mode} | 벤치마크: {benchmark_symbol}"
        )

        # 활성 페어 확인
        active_pairs = self._resolve_pairs()
        self.logger.info(f"활성 페어: {len(active_pairs)}개")
        for p in active_pairs:
            self.logger.info(f"  {p['name']}: {p['lead']} → {p['lag']}")

        if not active_pairs:
            raise ValueError("활성 페어 없음. US 데이터를 확인하세요.")

        # US 수익률 사전 계산 (날짜 → {심볼: 수익률})
        us_returns_by_date = self._precompute_us_returns(active_pairs)
        self.logger.info(f"US 수익률 데이터: {len(us_returns_by_date)}일")

        # KR 데이터 날짜 인덱싱
        bench_df = self._kr_data[benchmark_symbol]
        n_days = len(bench_df)
        warmup = max(90, self.regime_detector.config.get("lookback_days", 120))

        if n_days <= warmup:
            raise ValueError(f"데이터 부족: {n_days}일 (최소 {warmup}일)")

        # HMM 초기 학습
        bench_prices = bench_df["close"]
        self.regime_detector.fit_hmm(bench_prices.iloc[:warmup])

        cash = initial_cash
        positions: dict[str, BacktestPosition] = {}
        equity_history = []
        trades = []
        regime_pnl: dict[str, float] = {}
        pair_pnl: dict[str, float] = {}
        regime_day_counts: dict[str, int] = {}
        total_holding_days = 0
        total_closed = 0

        for i in range(warmup, n_days):
            date = bench_df["date"].iloc[i]

            # 레짐 감지
            regime_state = self.regime_detector.detect(bench_prices.iloc[:i + 1])
            regime = regime_state.regime
            regime_day_counts[regime.value] = regime_day_counts.get(regime.value, 0) + 1

            # 30일마다 HMM 재학습
            if i % 30 == 0 and i > warmup + 30:
                self.regime_detector.fit_hmm(bench_prices.iloc[:i + 1])

            # 날짜 필터
            if filter_start and date < filter_start:
                continue
            if filter_end and date > filter_end:
                continue

            # ── 청산 처리 ──
            self._process_exits(
                i, date, regime, regime_state, positions, trades,
                cash, regime_pnl, pair_pnl, total_holding_days, total_closed,
            )
            # process_exits가 positions/cash 등을 직접 수정하므로 locals에서 가져올 수 없음
            # 대신 mutable 객체로 처리

            # CRASH: 전량 청산 후 신규 진입 안 함
            if regime == Regime.CRASH:
                for sym in list(positions.keys()):
                    pos = positions[sym]
                    if sym not in self._kr_data or i >= len(self._kr_data[sym]):
                        continue
                    cur_price = self._kr_data[sym]["close"].iloc[i]
                    pnl = (cur_price - pos.avg_price) * pos.qty
                    cash += cur_price * pos.qty
                    regime_pnl["CRASH"] = regime_pnl.get("CRASH", 0) + pnl
                    pair_pnl[pos.pair_name] = pair_pnl.get(pos.pair_name, 0) + pnl
                    days_held = i - pos.entry_idx
                    total_holding_days += days_held
                    total_closed += 1
                    trades.append(BacktestTrade(
                        date=date, symbol=sym, side="sell", qty=pos.qty,
                        price=cur_price, regime="CRASH",
                        strategy="crash_exit", reason="CRASH 전면 매도",
                        pair_name=pos.pair_name,
                    ))
                    del positions[sym]

                self._record_equity(equity_history, date, cash, positions, i)
                continue

            # ── 시간/모드 기반 청산 ──
            for sym in list(positions.keys()):
                pos = positions[sym]
                if sym not in self._kr_data or i >= len(self._kr_data[sym]):
                    continue

                current_price = self._kr_data[sym]["close"].iloc[i]
                days_held = i - pos.entry_idx
                should_close = False
                reason = ""

                if self.exit_mode == "intraday":
                    # intraday: 진입 당일 종가에 청산
                    if days_held >= 1:
                        should_close = True
                        reason = "당일청산(intraday)"

                elif self.exit_mode == "next_close":
                    # next_close: 진입 다음날 종가에 청산
                    if days_held >= 2:
                        should_close = True
                        reason = "익일청산(next_close)"

                elif self.exit_mode == "swing":
                    # swing: ATR 트레일링 스탑 + 시간 제한
                    if current_price > pos.highest_price:
                        pos.highest_price = current_price
                        new_stop = pos.highest_price - (pos.atr_at_entry * pos.atr_stop_mult)
                        pos.trailing_stop = max(pos.trailing_stop, new_stop)

                    pnl_pct = (current_price - pos.avg_price) / pos.avg_price

                    if current_price <= pos.trailing_stop:
                        should_close = True
                        reason = f"ATR스탑 ({pnl_pct:+.2%})"
                    elif days_held >= pos.max_holding_days:
                        should_close = True
                        reason = f"보유기한 ({days_held}일)"

                if should_close:
                    pnl = (current_price - pos.avg_price) * pos.qty
                    cash += current_price * pos.qty
                    regime_pnl[regime.value] = regime_pnl.get(regime.value, 0) + pnl
                    pair_pnl[pos.pair_name] = pair_pnl.get(pos.pair_name, 0) + pnl
                    total_holding_days += days_held
                    total_closed += 1
                    trades.append(BacktestTrade(
                        date=date, symbol=sym, side="sell", qty=pos.qty,
                        price=current_price, regime=regime.value,
                        strategy=self.exit_mode, reason=reason,
                        pair_name=pos.pair_name,
                    ))
                    del positions[sym]

            # ── 신규 진입 ──
            # 전일(어제) KR 거래일에 대응되는 US 수익률 찾기
            us_rets = self._get_us_returns_for_kr_date(date, us_returns_by_date)

            if us_rets:
                held = set(positions.keys())
                signals = self.signal_generator.generate(
                    us_returns=us_rets,
                    regime_state=regime_state,
                    current_positions=held,
                )

                for sig in signals:
                    if sig.direction != "long":
                        continue
                    if sig.symbol not in self._kr_data or i >= len(self._kr_data[sig.symbol]):
                        continue
                    if sig.symbol in positions:
                        continue

                    # 시가에 매수
                    entry_price = self._kr_data[sig.symbol]["open"].iloc[i]
                    if entry_price <= 0:
                        continue

                    # intraday 모드: 시가에 매수, 당일 종가에 매도
                    if self.exit_mode == "intraday":
                        close_price = self._kr_data[sig.symbol]["close"].iloc[i]
                        total_val = cash  # 간단히 현금만으로 계산
                        qty = self._calc_qty(
                            cash, total_val, entry_price,
                            sig.strength, regime_state.position_multiplier,
                            0, 0,
                        )
                        if qty > 0 and cash >= entry_price * qty:
                            cash -= entry_price * qty
                            pnl = (close_price - entry_price) * qty
                            cash += close_price * qty
                            regime_pnl[regime.value] = regime_pnl.get(regime.value, 0) + pnl
                            pair_pnl[sig.pair_name] = pair_pnl.get(sig.pair_name, 0) + pnl
                            total_holding_days += 0  # 당일
                            total_closed += 1
                            trades.append(BacktestTrade(
                                date=date, symbol=sig.symbol, side="buy", qty=qty,
                                price=entry_price, regime=regime.value,
                                strategy="gap_intraday", reason=sig.reason,
                                pair_name=sig.pair_name,
                            ))
                            trades.append(BacktestTrade(
                                date=date, symbol=sig.symbol, side="sell", qty=qty,
                                price=close_price, regime=regime.value,
                                strategy="gap_intraday", reason="당일청산",
                                pair_name=sig.pair_name,
                            ))
                        continue

                    # next_close / swing: 포지션으로 관리
                    total_val = cash + sum(
                        self._kr_data[s]["close"].iloc[min(i, len(self._kr_data[s]) - 1)] * p.qty
                        for s, p in positions.items()
                        if s in self._kr_data and i < len(self._kr_data[s])
                    )

                    # 익스포저 체크
                    invested = sum(
                        self._kr_data[s]["close"].iloc[min(i, len(self._kr_data[s]) - 1)] * p.qty
                        for s, p in positions.items()
                        if s in self._kr_data and i < len(self._kr_data[s])
                    )
                    if total_val > 0 and invested / total_val >= self.max_exposure:
                        continue

                    atr_val = self._calc_atr_at(sig.symbol, i)
                    atr_mult = regime_state.atr_stop_multiplier

                    qty = self._calc_qty(
                        cash, total_val, entry_price,
                        sig.strength, regime_state.position_multiplier,
                        atr_val, atr_mult,
                    )

                    if qty > 0 and cash >= entry_price * qty:
                        cash -= entry_price * qty
                        stop_level = (
                            entry_price - (atr_val * atr_mult)
                            if atr_val > 0
                            else entry_price * 0.95
                        )

                        positions[sig.symbol] = BacktestPosition(
                            symbol=sig.symbol, qty=qty, avg_price=entry_price,
                            entry_idx=i, pair_name=sig.pair_name,
                            atr_at_entry=atr_val, highest_price=entry_price,
                            trailing_stop=stop_level, atr_stop_mult=atr_mult,
                            max_holding_days=self.max_holding_days,
                        )
                        trades.append(BacktestTrade(
                            date=date, symbol=sig.symbol, side="buy", qty=qty,
                            price=entry_price, regime=regime.value,
                            strategy=f"gap_{self.exit_mode}", reason=sig.reason,
                            pair_name=sig.pair_name,
                        ))

            self._record_equity(equity_history, date, cash, positions, i)

        # ── 결과 계산 ──
        if not equity_history:
            raise ValueError("시뮬레이션 결과 없음")

        equity_df = pd.DataFrame(equity_history)
        equity_series = equity_df.set_index("date")["equity"]

        total_return = (equity_series.iloc[-1] / initial_cash) - 1
        days = max((equity_series.index[-1] - equity_series.index[0]).days, 1)
        annualized = (1 + total_return) ** (365 / days) - 1

        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()

        daily_returns = equity_series.pct_change().dropna()
        sharpe = (
            float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
            if daily_returns.std() > 0 else 0
        )

        sell_trades = [t for t in trades if t.side == "sell"]
        wins = 0
        gross_profit = 0.0
        gross_loss = 0.0
        for st in sell_trades:
            entry = self._find_entry_price(st.symbol, trades, st)
            pnl = (st.price - entry) * st.qty
            if pnl > 0:
                wins += 1
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)

        win_rate = wins / len(sell_trades) if sell_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_holding = total_holding_days / total_closed if total_closed > 0 else 0

        total_regime_days = sum(regime_day_counts.values())
        regime_dist = {
            k: v / total_regime_days for k, v in regime_day_counts.items()
        } if total_regime_days > 0 else {}

        result = BacktestResult(
            total_return=total_return,
            annualized_return=annualized,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_holding_days=avg_holding,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_series,
            regime_returns={k: v / initial_cash for k, v in regime_pnl.items()},
            pair_returns={k: v / initial_cash for k, v in pair_pnl.items()},
            regime_distribution=regime_dist,
        )

        self.logger.info(f"\n{result.summary()}")
        return result

    # ──────────────────────────────────────────────
    # US 수익률 계산
    # ──────────────────────────────────────────────

    def _precompute_us_returns(
        self, active_pairs: list[dict]
    ) -> dict[pd.Timestamp, dict[str, float]]:
        """
        US 데이터에서 일별 수익률 사전 계산.
        반환: {US날짜: {심볼: 전일대비수익률}}
        """
        result: dict[pd.Timestamp, dict[str, float]] = {}

        lead_symbols = set()
        for pair in active_pairs:
            lead_symbols.add(pair["lead"])

        for sym in lead_symbols:
            if sym not in self._us_data:
                continue
            df = self._us_data[sym]
            closes = df["close"].values
            dates = df["date"].values

            for j in range(1, len(df)):
                if closes[j - 1] <= 0:
                    continue
                ret = (closes[j] / closes[j - 1]) - 1
                dt = pd.Timestamp(dates[j])
                if dt not in result:
                    result[dt] = {}
                result[dt][sym] = ret

        return result

    def _get_us_returns_for_kr_date(
        self, kr_date: pd.Timestamp,
        us_returns_by_date: dict[pd.Timestamp, dict[str, float]],
    ) -> dict[str, float]:
        """
        KR 거래일에 대응되는 US 전일 수익률 찾기.
        KR 거래일 기준 직전 US 거래일의 수익률 반환.

        로직: kr_date 이전 최근 US 거래일의 수익률을 사용.
        (US는 보통 KR보다 1영업일 앞서거나 같은 날)
        """
        # kr_date 당일 또는 직전 1~5일 중 US 데이터가 있는 가장 최근 날짜
        for offset in range(0, 6):
            check_date = kr_date - pd.Timedelta(days=offset)
            if check_date in us_returns_by_date:
                return us_returns_by_date[check_date]

        return {}

    # ──────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────

    def _resolve_pairs(self) -> list[dict]:
        """사용 가능한 페어만 필터"""
        active = []
        for pair in self.pair_configs:
            lead_sym = pair["lead"]
            if lead_sym not in self._us_data:
                continue

            available_lags = [s for s in pair["lag"] if s in self._kr_data]
            if not available_lags:
                continue

            active.append({
                "name": pair.get("name", f"{lead_sym}→KR"),
                "lead": lead_sym,
                "lag": available_lags,
            })
        return active

    def _process_exits(self, i, date, regime, regime_state, positions, trades,
                       cash, regime_pnl, pair_pnl, total_holding_days, total_closed):
        """공통 청산 처리 (swing 모드에서 ATR 스탑용)"""
        # swing 모드가 아니면 별도 처리 불필요 (메인 루프에서 처리)
        pass

    def _calc_atr_at(self, symbol: str, idx: int) -> float:
        if symbol not in self._kr_data:
            return 0.0
        df = self._kr_data[symbol]
        end = min(idx + 1, len(df))
        if end < 15:
            return 0.0
        start = max(0, end - 30)
        sliced = df.iloc[start:end]
        atr_series = calc_atr(sliced["high"], sliced["low"], sliced["close"], 14)
        val = atr_series.iloc[-1]
        return float(val) if not pd.isna(val) else 0.0

    def _calc_qty(
        self,
        cash: float,
        total: float,
        price: float,
        signal_strength: float,
        regime_mult: float,
        atr: float,
        atr_mult: float,
    ) -> int:
        if price <= 0 or total <= 0 or regime_mult <= 0:
            return 0

        base = total * self.max_position_pct * self.kelly_fraction
        adjusted = base * regime_mult * max(0.3, min(signal_strength, 1.0))

        if atr > 0 and atr_mult > 0:
            risk_per_share = atr * atr_mult
            atr_qty = adjusted / risk_per_share
        else:
            atr_qty = adjusted / price

        cash_qty = (cash * 0.9) / price
        qty = int(min(atr_qty, cash_qty))
        return max(0, qty)

    def _record_equity(
        self, history: list, date, cash: float,
        positions: dict[str, BacktestPosition], idx: int,
    ):
        value = cash
        for sym, pos in positions.items():
            if sym in self._kr_data and idx < len(self._kr_data[sym]):
                value += self._kr_data[sym]["close"].iloc[idx] * pos.qty
        history.append({"date": date, "equity": value})

    @staticmethod
    def _find_entry_price(
        symbol: str, trades: list[BacktestTrade], sell_trade: BacktestTrade
    ) -> float:
        for t in reversed(trades):
            if t.symbol == symbol and t.side == "buy" and t.date <= sell_trade.date:
                return t.price
        return sell_trade.price
