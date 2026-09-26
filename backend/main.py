import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import datasets

app = FastAPI(
    title="EntityMatch AI Backend",
    description="Production Business Entity Resolution Platform API",
    version="1.0.0"
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "EntityMatch AI — Business Entity Resolution API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
