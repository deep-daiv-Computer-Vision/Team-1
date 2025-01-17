import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv('env/.env')


class Settings(BaseSettings):
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    MONGO_DB_URI: str = os.getenv("MONGO_DB_URI")

settings = Settings()
