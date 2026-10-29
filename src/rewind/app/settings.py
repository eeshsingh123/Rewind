from decimal import Decimal
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

HARD_LIMIT_USD = Decimal("140")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REWIND_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Optional so the app starts without it; adapters that call Meta fail at wiring instead.
    meta_api_key: SecretStr | None = Field(default=None, validation_alias="META_API_KEY")
    meta_api_base_url: str = "https://api.meta.ai/v1"
    # Uploaded clips expire on Meta's side after this long (Meta allows 3600 to 2592000).
    meta_file_ttl_s: int = Field(default=86_400, ge=3600, le=2_592_000)
    data_root: Path = Path("data")
    muse_model: str = "muse-spark-1.3-contributor"
    # Base of the signed media URLs; the cloudflared tunnel URL when Meta must fetch them.
    public_base_url: str = "http://127.0.0.1:8000"
    # Without one, a random key is made at start-up, so signed URLs die on restart.
    media_signing_key: SecretStr | None = None
    # The proxy fps is the SAM cost knob: every frame of the proxy is billed per concept.
    track_fps: float = Field(default=2.0, gt=0)
    proxy_max_height: int = Field(default=720, gt=0)
    workspace_daily_cap_usd: Decimal = Field(default=Decimal("5"), ge=0)
    # All-time spend across workspaces. The account has $150 of credit, so the setting may be
    # raised for a demo but never past HARD_LIMIT_USD, which keeps the card from being charged.
    kill_switch_usd: Decimal = Field(default=Decimal("120"), ge=0, le=HARD_LIMIT_USD)
    # SQLite under the data root until Neon; the same code runs on a Postgres URL.
    database_url: str | None = None
    # Chat channels: a channel whose token is unset is not started. Ids are comma-separated.
    telegram_bot_token: SecretStr | None = None
    telegram_allowed_user_ids: Annotated[set[str], NoDecode] = Field(default_factory=set[str])
    whatsapp_token: SecretStr | None = None
    whatsapp_phone_number_id: str | None = None
    whatsapp_app_id: str | None = None
    whatsapp_app_secret: SecretStr | None = None
    whatsapp_verify_token: SecretStr | None = None
    # Comma-separated, in international format without the plus.
    whatsapp_allowed_numbers: Annotated[set[str], NoDecode] = Field(default_factory=set[str])
    # An analysis estimated above this asks for a confirm button first (D-7).
    confirm_above_usd: Decimal = Field(default=Decimal("0.25"), ge=0)

    @property
    def database_url_or_default(self) -> str:
        path = (self.data_root / "rewind.db").as_posix()
        return self.database_url or f"sqlite+aiosqlite:///{path}"

    @field_validator("telegram_allowed_user_ids", "whatsapp_allowed_numbers", mode="before")
    @classmethod
    def _split_list(cls, value: object) -> object:
        if isinstance(value, str):
            return {item.strip().lstrip("+") for item in value.split(",") if item.strip()}
        return value
