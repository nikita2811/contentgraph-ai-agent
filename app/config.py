from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    google_api_key: str
    gemini_model: str = "gemini-1.5-flash"
    tavily_api_key:str
    serpapi_api_key:str

    class Config:
        env_file = ".env"

settings = Settings()