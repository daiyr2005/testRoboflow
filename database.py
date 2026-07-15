from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from config import config
from model import PredictionModel

client = AsyncIOMotorClient(
    config.MONGODB_URL
)

database = client[config.DB_NAME]

async  def init_db():
   await init_beanie(database=database, document_models=[PredictionModel])

try:
    client.server_info()
    print('База данных уланды')
except Exception as e:
    print(f'Ошипка, {e}')





