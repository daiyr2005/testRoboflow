from logging import config

from dotenv import load_dotenv
import os

load_dotenv()
class Config:
    MONGODB_URL = os.getenv('MONGODB_URL')
    DB_NAME = os.getenv('DB_NAME')
    ROBOFLOW_URL = os.getenv('ROBOFLOW_URL')
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')


config = Config()
