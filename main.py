from fastapi import FastAPI
import uvicorn
from database import init_db
from contextlib import asynccontextmanager
from router import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


check_image_app = FastAPI()
check_image_app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(check_image_app, host='127.0.0.1', port = 8000)