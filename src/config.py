from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-2.0-flash"
    chunk_size: int = 1000  
    chunk_overlap: int = 200  
    top_k_results: int = 5   

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
