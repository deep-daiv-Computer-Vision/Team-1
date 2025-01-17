from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import database, qwen


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(database.router)
app.include_router(qwen.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
