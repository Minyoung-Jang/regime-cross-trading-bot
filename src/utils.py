"""
utils.py - 유틸리티 함수 모음
"""
import logging
import requests
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # 파일 핸들러
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(
            log_dir / f"bot_{datetime.now(KST).strftime('%Y%m%d')}.log",
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def send_telegram(token: str, chat_id: str, message: str):
    """텔레그램 봇 알림 전송"""
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": f"[{datetime.now(KST).strftime('%H:%M:%S')}] {message}",
        }
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass


def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) 계산"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_bollinger_bands(
    prices: pd.Series, period: int = 20, std_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """볼린저 밴드 계산 → (상단, 중간, 하단)"""
    mid = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def calc_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """ATR (Average True Range) 계산"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calc_returns(prices: pd.Series) -> pd.Series:
    """로그 수익률 계산"""
    return np.log(prices / prices.shift(1)).dropna()


def calc_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """롤링 변동성 계산"""
    returns = calc_returns(prices)
    return returns.rolling(window=window).std() * np.sqrt(252)


def is_kr_market_open() -> bool:
    """한국 시장 개장 여부 확인 (09:00 ~ 15:30 KST)"""
    now = datetime.now(KST)
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    weekday = now.weekday()
    return weekday < 5 and market_open <= now <= market_close


def is_us_market_open() -> bool:
    """미국 시장 개장 여부 확인 (NYSE: 09:30 ~ 16:00 ET, 서머타임 자동 반영)"""
    now = datetime.now(ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    weekday = now.weekday()
    return weekday < 5 and market_open <= now <= market_close


def format_krw(amount: float) -> str:
    """금액을 한국 원화 형식으로 포맷"""
    if amount >= 1_0000_0000:
        return f"{amount / 1_0000_0000:.1f}억원"
    elif amount >= 1_0000:
        return f"{amount / 1_0000:.0f}만원"
    else:
        return f"{amount:,.0f}원"


def calc_realized_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """롤링 실현 변동성 (연환산)"""
    returns = calc_returns(prices)
    return returns.rolling(window=window).std() * np.sqrt(252)


def calc_vol_of_vol(prices: pd.Series, vol_window: int = 20, vov_window: int = 20) -> float:
    """변동성의 변동성 (계수 of 변동: std(vol) / mean(vol))"""
    rolling_vol = calc_realized_volatility(prices, vol_window)
    recent_vol = rolling_vol.dropna().iloc[-vov_window:]
    if len(recent_vol) < vov_window // 2:
        return 0.0
    mean_vol = recent_vol.mean()
    if mean_vol == 0:
        return 0.0
    return float(recent_vol.std() / mean_vol)


def calc_sma(prices: pd.Series, window: int) -> pd.Series:
    """단순 이동 평균"""
    return prices.rolling(window=window).mean()
