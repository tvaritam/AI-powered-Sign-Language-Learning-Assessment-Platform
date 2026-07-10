from fastapi import FastAPI
from app.api.endpoints import preprocess  # Import the preprocess endpoint

# 1. First, create the FastAPI app instance!
app = FastAPI(title="Sign Language Platform API")

# 2. Now you can safely include your routers
app.include_router(
    preprocess.router, 
    prefix="/api/v1",  
    tags=["Dataset Preprocessing Pipeline Admin"]
)

# Your other code or routers below...