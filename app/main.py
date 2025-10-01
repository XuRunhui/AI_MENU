"""Main FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time
from loguru import logger

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1.api import api_router

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Menu Taste Guide API",
    description="AI-powered menu recommendation system with taste-centric descriptions",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.ENV != "production" else None,
    docs_url=f"{settings.API_V1_STR}/docs" if settings.ENV != "production" else None,
    redoc_url=f"{settings.API_V1_STR}/redoc" if settings.ENV != "production" else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENV == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.ENV == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure based on your domain
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.3f}s")

    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)

    return response


# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Root endpoint for health checks."""
    return {
        "message": "Menu Taste Guide API",
        "version": "1.0.0",
        "status": "healthy",
        "environment": settings.ENV
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "environment": settings.ENV
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )