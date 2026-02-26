"""
main.py - Regime-Gated Gap Trading 봇

실행 모드:
  1. 라이브 모의투자:  python main.py
  2. 백테스트:         python main.py --backtest --start 2024-01-01 --end 2024-12-31
  3. 상태 확인:        python main.py --status
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import signal
from datetime import datetime, timedelta

import schedule

from src.config import load_config
from src.utils import setup_logger, is_kr_market_open, send_telegram, format_krw
from src.kis_trader import KisTrader
from src.strategy import RegimeGatedStrategy
from src.backtest import Backtester


# ──────────────────────────────────────────────
# 글로벌 설정
# ──────────────────────────────────────────────

CYCLE_INTERVAL_MINUTES = 5  # 전략 사이클 실행 간격 (분)
DAILY_REPORT_TIME = "15:40" # 일일 리포트 시간 (장 마감 후)
BOT_VERSION = "2.0.0"

running = True


def signal_handler(signum, frame):
    """Ctrl+C 핸들러"""
    global running
    running = False
    print("\n\n봇 종료 요청됨. 안전하게 종료합니다...")


# ──────────────────────────────────────────────
# 라이브 모드
# ──────────────────────────────────────────────

def run_live(config: dict):
    """라이브 모의투자/실전투자 모드"""
    logger = setup_logger("Main")
    logger.info(f"Trading Bot v{BOT_VERSION} 시작")
    logger.info(f"   모드: {'모의투자' if config['kis']['is_virtual'] else '실전투자'}")

    # 초기화
    trader = KisTrader(config)
    strategy = RegimeGatedStrategy(config, trader)

    noti = config.get("notification", {})
    tg_token = noti.get("telegram_bot_token", "")
    tg_chat = noti.get("telegram_chat_id", "")

    # 시작 알림
    balance = trader.get_balance()
    logger.info(
        f"   계좌: 현금 {format_krw(balance.get('cash', 0))} | "
        f"총평가 {format_krw(balance.get('total', 0))}"
    )

    send_telegram(tg_token, tg_chat, f"봇 시작 | 총평가: {format_krw(balance.get('total', 0))}")

    # ── 전략 사이클 함수 ──
    def run_cycle():
        if not is_kr_market_open():
            return

        try:
            logger.info("-" * 40)
            logger.info(f"전략 사이클 실행 ({datetime.now().strftime('%H:%M:%S')})")
            decisions = strategy.run_cycle()

            if decisions:
                logger.info(f"이번 사이클 결정: {len(decisions)}건")
                for d in decisions:
                    logger.info(
                        f"  {d.action:4s} {d.symbol} {d.qty}주 | "
                        f"[{d.strategy}] {d.reason}"
                    )
            else:
                logger.info("매매 결정 없음")

        except Exception as e:
            logger.error(f"사이클 실행 오류: {e}", exc_info=True)

    # ── 일일 리포트 함수 ──
    def daily_report():
        try:
            report = strategy.get_status_report()
            logger.info(f"\n{report}")

            send_telegram(tg_token, tg_chat, report)

            trader.reset_daily_count()
            strategy.risk_manager.reset_daily()

        except Exception as e:
            logger.error(f"일일 리포트 오류: {e}")

    # ── 한국장 개장 전 US 수익률 분석 ──
    def pre_market_analysis():
        try:
            logger.info("한국장 개장 전 US 수익률 분석...")
            report = strategy.get_pre_market_report()
            logger.info(f"\n{report}")

            send_telegram(tg_token, tg_chat, f"Gap Trading 분석\n{report}")

        except Exception as e:
            logger.error(f"프리마켓 분석 오류: {e}")

    # ── 스케줄 등록 ──
    schedule.every(CYCLE_INTERVAL_MINUTES).minutes.do(run_cycle)
    schedule.every().day.at(DAILY_REPORT_TIME).do(daily_report)
    schedule.every().day.at("08:30").do(pre_market_analysis)

    logger.info(
        f"스케줄 등록 완료:\n"
        f"   - 전략 사이클: {CYCLE_INTERVAL_MINUTES}분 간격\n"
        f"   - 프리마켓 분석: 매일 08:30\n"
        f"   - 일일 리포트: 매일 {DAILY_REPORT_TIME}\n"
        f"   - Ctrl+C로 종료"
    )

    # ── 메인 루프 ──
    while running:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            break

    # 종료 처리
    logger.info("봇 종료")
    send_telegram(tg_token, tg_chat, "봇 종료됨")


# ──────────────────────────────────────────────
# 백테스트 모드
# ──────────────────────────────────────────────

def run_backtest(config: dict, start: str, end: str):
    """백테스트 모드"""
    logger = setup_logger("Backtest")
    logger.info(f"백테스트 모드: {start} ~ {end}")

    bt = Backtester(config)

    data_dir = "data"
    if not os.path.exists(data_dir):
        logger.error(
            f"데이터 디렉토리 '{data_dir}/'가 없습니다.\n"
            f"아래 형식의 CSV 파일을 준비하세요:\n"
            f"  data/005930.csv  (컬럼: date, open, high, low, close, volume)\n"
            f"  data/000660.csv\n"
            f"  ...\n"
            f"\n"
            f"Yahoo Finance 등에서 다운로드 가능합니다."
        )
        return

    filepaths = {}
    for f in os.listdir(data_dir):
        if f.endswith(".csv"):
            symbol = f.replace(".csv", "")
            filepaths[symbol] = os.path.join(data_dir, f)

    if not filepaths:
        logger.error("데이터 파일이 없습니다.")
        return

    logger.info(f"데이터 파일: {list(filepaths.keys())}")
    bt.load_csv(filepaths)

    # US 데이터 로드 (크로스마켓 백테스트)
    us_data_dir = os.path.join(data_dir, "us")
    if os.path.exists(us_data_dir):
        us_filepaths = {}
        for f in os.listdir(us_data_dir):
            if f.endswith(".csv"):
                symbol = f.replace(".csv", "")
                us_filepaths[symbol] = os.path.join(us_data_dir, f)
        if us_filepaths:
            bt.load_us_csv(us_filepaths)
        else:
            logger.warning("US 데이터 파일 없음 → 크로스마켓 비활성")
    else:
        logger.info("data/us/ 없음 → 크로스마켓 없이 레짐 전략만 실행")

    result = bt.run(initial_cash=500_000_000, start_date=start, end_date=end)
    print(result.summary())


# ──────────────────────────────────────────────
# 상태 확인 모드
# ──────────────────────────────────────────────

def show_status(config: dict):
    """현재 상태만 확인"""
    trader = KisTrader(config)
    strategy = RegimeGatedStrategy(config, trader)

    strategy._update_regime()

    print(strategy.get_status_report())
    print()
    print(strategy.get_pre_market_report())


# ──────────────────────────────────────────────
# 엔트리 포인트
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regime-Gated Gap Trading 봇")
    parser.add_argument("--backtest", action="store_true", help="백테스트 모드")
    parser.add_argument("--start", default="2024-01-01", help="백테스트 시작일")
    parser.add_argument("--end", default="2024-12-31", help="백테스트 종료일")
    parser.add_argument("--status", action="store_true", help="현재 상태 확인")

    args = parser.parse_args()

    # 설정 로드 (.env + config.yaml 자동 병합)
    config = load_config()

    # 시그널 핸들러
    signal.signal(signal.SIGINT, signal_handler)
    try:
        signal.signal(signal.SIGTERM, signal_handler)  # 윈도우 미지원
    except (OSError, AttributeError):
        pass

    if args.backtest:
        run_backtest(config, args.start, args.end)
    elif args.status:
        show_status(config)
    else:
        run_live(config)


if __name__ == "__main__":
    main()
