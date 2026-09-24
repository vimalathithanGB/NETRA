from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _BACKEND_DIR / ".env"


class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "NETRA Traffic Intelligence Backend"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Database (PostgreSQL + PostGIS)
    DATABASE_URL: str = "postgresql://postgres:postgres@127.0.0.1:5432/netra_db"

    # Security & Tokens
    SECRET_KEY: str = "CHANGE_THIS_IN_DEVELOPMENT"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"

    # API Routing
    API_V1_PREFIX: str = "/api/v1"

    # CORS
    CORS_ORIGINS: Union[List[str], str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=(str(_ENV_FILE), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )



settings = Settings()
