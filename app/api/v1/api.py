"""API v1 router that includes all endpoint modules."""

from fastapi import APIRouter

from app.api.v1.endpoints import ocr, places, taste_cards, combos

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(ocr.router, prefix="/ocr", tags=["OCR"])
api_router.include_router(places.router, prefix="/places", tags=["Places"])
api_router.include_router(taste_cards.router, prefix="/taste-cards", tags=["Taste Cards"])
api_router.include_router(combos.router, prefix="/combos", tags=["Combos"])