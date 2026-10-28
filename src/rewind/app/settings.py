from decimal import Decimal
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    kill_switch_usd: Decimal = Field(default=Decimal("120"), ge=0)
