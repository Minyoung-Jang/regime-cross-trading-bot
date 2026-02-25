"""
config.py - 설정 로더

.env에서 민감 정보(API 키, 시크릿 등)를 로드하고
config.yaml에서 전략/리스크 설정을 로드하여 통합 config dict를 반환.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
import yaml


def load_config(yaml_path: str = "config.yaml") -> dict:
    """
    .env + config.yaml을 합쳐 통합 설정 딕셔너리 반환.

    Returns:
        dict with keys: kis, strategy, risk, notification
    """
    # .env 로드 (프로젝트 루트 기준)
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    # config.yaml 로드
    yaml_full_path = project_root / yaml_path
    with open(yaml_full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # KIS 인증 정보 (.env → config["kis"])
    is_virtual = os.getenv("IS_VIRTUAL", "true").lower() == "true"
    config["kis"] = {
        "id": os.getenv("HTS_ID", ""),
        "account": os.getenv("VIRTUAL_ACCOUNT_NUMBER", "") if is_virtual else os.getenv("ACCOUNT_NUMBER", ""),
        "appkey": os.getenv("API_KEY", ""),
        "secretkey": os.getenv("SECRET_KEY", ""),
        "virtual_id": os.getenv("HTS_ID", ""),
        "virtual_appkey": os.getenv("VIRTUAL_API_KEY", ""),
        "virtual_secretkey": os.getenv("VIRTUAL_SECRET_KEY", ""),
        "is_virtual": is_virtual,
    }

    # 알림 설정 (.env → config["notification"])
    config.setdefault("notification", {})
    config["notification"]["discord_webhook"] = os.getenv("DISCORD_WEBHOOK", "")
    config["notification"]["telegram_bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN", "")
    config["notification"]["telegram_chat_id"] = os.getenv("TELEGRAM_CHAT_ID", "")
    config["notification"]["log_level"] = os.getenv(
        "LOG_LEVEL",
        config["notification"].get("log_level", "INFO"),
    )

    return config
