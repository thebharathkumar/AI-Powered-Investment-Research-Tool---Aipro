from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-preview"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-sonnet-20240229"

    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/aipro"
    sync_database_url: str = "postgresql://user:password@localhost:5432/aipro"

    # Vector Store
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "financial_documents"

    # AWS
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket_name: str = "aipro-documents"

    # App
    app_env: str = "development"
    secret_key: str = "changeme"
    api_prefix: str = "/api/v1"
    debug: bool = True
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
