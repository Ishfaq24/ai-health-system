from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="AI Health Detection API")

app.include_router(router)