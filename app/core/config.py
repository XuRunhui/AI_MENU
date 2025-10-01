"""Application configuration settings."""

from typing import Optional
from pydantic import validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Google Services
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GOOGLE_PLACES_API_KEY: str

    # Yelp API
    YELP_API_KEY: str

    # OpenAI
    OPENAI_API_KEY: str

    # App Settings
    ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    API_V1_STR: str = "/api/v1"

    # Cache Settings
    CACHE_TTL_HOURS: int = 24
    REVIEW_FETCH_LIMIT: int = 100

    # ML Settings
    SIMILARITY_THRESHOLD: float = 0.56
    MAX_REVIEW_SNIPPETS: int = 12

    # Model Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.3

    @validator("DATABASE_URL", pre=True)
    def validate_database_url(cls, v: str) -> str:
        """Ensure database URL is provided."""
        if not v:
            raise ValueError("DATABASE_URL must be provided")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()