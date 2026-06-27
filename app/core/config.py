"""
app/core/config.py

Application-wide settings loaded from the .env file via pydantic-settings.
All values have safe defaults so the server can start without a .env file
(useful for testing), but production deployments MUST override SECRET_KEY.

Note on pydantic-settings 2.14+:
  List fields are decoded as JSON by default. To avoid breakage with
  simple comma-separated .env values, we store ALLOWED_ORIGINS as a plain
  str and split it ourselves in a model_validator.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "DermAI Backend"
    app_version: str = "1.0.0"
    debug: bool = True
    # Stored as a comma-separated string in .env to avoid pydantic-settings
    # 2.14+ JSON-first decoding of list fields.
    # e.g.  ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"
    _allowed_origins_list: list[str] = []

    # ── Security ─────────────────────────────────────────────────────────────
    secret_key: str = "change-me-in-production-use-secrets-token-hex-32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./dermai.db"

    # ── ML Model ─────────────────────────────────────────────────────────────
    model_name: str = "google/vit-base-patch16-224"
    hf_cache_dir: str = "./model_cache"

    # ── File Storage ─────────────────────────────────────────────────────────
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 10

    @model_validator(mode="after")
    def parse_origins(self) -> "Settings":
        """Split the comma-separated ALLOWED_ORIGINS string into a list."""
        raw = self.allowed_origins
        if isinstance(raw, list):
            self._allowed_origins_list = raw
        else:
            self._allowed_origins_list = [o.strip() for o in raw.split(",") if o.strip()]
        return self

    def get_allowed_origins(self) -> list[str]:
        """Return the parsed list of allowed CORS origins."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_cache_path(self) -> Path:
        path = Path(self.hf_cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — safe to call anywhere."""
    return Settings()
