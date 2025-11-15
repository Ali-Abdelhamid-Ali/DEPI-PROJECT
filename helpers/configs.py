from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str 
    APP_VERSION: str 

    GENERATION_BACKEND: str 

    OPENAI_API_KEY: str 
    OPENAI_MODEL: str 
    OPENAI_EMBEDDING_MODEL: str 

    COHERE_API_KEY: str 
    COHERE_MODEL: str
    COHERE_EMBEDDING_MODEL: str

    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str

        
    INPUT_DAFAULT_MAX_CHARACTERS : int =1024
    GENERATION_DAFAULT_MAX_TOKENS : int=200
    GENERATION_DAFAULT_TEMPERATURE : float =0.1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings(): ## this makes any got by "get_settings().APP_NAME" 
    return Settings()