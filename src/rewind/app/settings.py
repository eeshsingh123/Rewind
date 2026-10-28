from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REWIND_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Optional so the app starts without it; adapters that call Meta fail at wiring instead.
    meta_api_key: SecretStr | None = Field(default=None, validation_alias="META_API_KEY")
    data_root: Path = Path("data")
