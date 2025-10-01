"""Taste card endpoints for dish descriptions."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from uuid import UUID

from app.db.base import get_db
from app.schemas.taste_card import TasteCardRequest, TasteCardResponse
from app.services.taste_card_service import TasteCardService
from app.services.place_service import PlaceService

router = APIRouter()


@router.post("/generate", response_model=TasteCardResponse)
async def generate_taste_card(
    taste_request: TasteCardRequest,
    db: AsyncSession = Depends(get_db)
) -> TasteCardResponse:
    """
    Generate a taste description card for a specific dish at a restaurant.

    This endpoint:
    1. Finds review mentions of the dish using semantic similarity
    2. Extracts taste aspects (spice, texture, richness) using ABSA
    3. Generates human-readable taste bullets using LLM
    4. Provides confidence levels and source attribution

    Args:
        taste_request: Dish and restaurant details
        db: Database session

    Returns:
        TasteCardResponse with taste description and metadata

    Example:
        ```python
        request_data = {
            "place_id": "uuid-123",
            "dish_name": "Mapo Tofu",
            "force_regenerate": false
        }
        response = requests.post("/api/v1/taste-cards/generate", json=request_data)

        # Response includes detailed taste analysis
        {
            "dish_name": "Mapo Tofu",
            "bullets": [
                "Intensely spicy with tongue-numbing Sichuan peppercorns",
                "Silky soft tofu in rich, oily red sauce",
                "High umami from ground pork and fermented beans"
            ],
            "aspects": {
                "spice": 3,
                "heat_type": ["peppercorn", "chili"],
                "texture": ["silky", "soft"],
                "richness": "heavy"
            },
            "confidence_level": "high",
            "fallback_level": "A",
            "sources": ["120+ Google reviews", "85 Yelp reviews"]
        }
        ```
    """
    try:
        logger.info(f"Generating taste card for dish '{taste_request.dish_name}' at place {taste_request.place_id}")

        # Initialize services
        taste_card_service = TasteCardService(db)
        place_service = PlaceService(db)

        # Verify place exists
        place = await place_service.get_place_by_id(taste_request.place_id)
        if not place:
            raise HTTPException(
                status_code=404,
                detail=f"Place {taste_request.place_id} not found"
            )

        # Check if taste card already exists (unless force regenerate)
        if not taste_request.force_regenerate:
            existing_card = await taste_card_service.get_existing_taste_card(
                place_id=taste_request.place_id,
                dish_name=taste_request.dish_name
            )
            if existing_card:
                logger.info(f"Returning existing taste card for {taste_request.dish_name}")
                return existing_card.to_api_response()

        # Generate new taste card
        taste_card = await taste_card_service.generate_taste_card(
            place=place,
            dish_name=taste_request.dish_name
        )

        logger.info(f"Generated taste card with confidence: {taste_card.confidence_level}")

        return taste_card.to_api_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating taste card: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate taste card: {str(e)}"
        )


@router.get("/{place_id}", response_model=List[TasteCardResponse])
async def get_place_taste_cards(
    place_id: UUID,
    limit: int = Query(50, ge=1, le=100, description="Maximum cards to return"),
    confidence_min: str = Query("low", description="Minimum confidence level (low, medium, high)"),
    db: AsyncSession = Depends(get_db)
) -> List[TasteCardResponse]:
    """
    Get all taste cards for a specific restaurant.

    Args:
        place_id: UUID of the restaurant
        limit: Maximum number of cards to return
        confidence_min: Minimum confidence level filter
        db: Database session

    Returns:
        List of taste cards for the restaurant
    """
    try:
        logger.info(f"Getting taste cards for place {place_id}")

        taste_card_service = TasteCardService(db)

        # Verify place exists
        place_service = PlaceService(db)
        place = await place_service.get_place_by_id(place_id)
        if not place:
            raise HTTPException(
                status_code=404,
                detail=f"Place {place_id} not found"
            )

        # Get taste cards
        taste_cards = await taste_card_service.get_place_taste_cards(
            place_id=place_id,
            limit=limit,
            confidence_min=confidence_min
        )

        return [card.to_api_response() for card in taste_cards]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting taste cards for place {place_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get taste cards: {str(e)}"
        )


@router.get("/{place_id}/{dish_name}", response_model=TasteCardResponse)
async def get_specific_taste_card(
    place_id: UUID,
    dish_name: str,
    db: AsyncSession = Depends(get_db)
) -> TasteCardResponse:
    """
    Get taste card for a specific dish at a restaurant.

    Args:
        place_id: UUID of the restaurant
        dish_name: Name of the dish
        db: Database session

    Returns:
        Taste card for the specific dish
    """
    try:
        logger.info(f"Getting taste card for '{dish_name}' at place {place_id}")

        taste_card_service = TasteCardService(db)

        # Get specific taste card
        taste_card = await taste_card_service.get_existing_taste_card(
            place_id=place_id,
            dish_name=dish_name
        )

        if not taste_card:
            raise HTTPException(
                status_code=404,
                detail=f"Taste card for '{dish_name}' at place {place_id} not found"
            )

        return taste_card.to_api_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting specific taste card: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get taste card: {str(e)}"
        )


@router.delete("/{place_id}/{dish_name}")
async def delete_taste_card(
    place_id: UUID,
    dish_name: str,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Delete a specific taste card (for regeneration).

    Args:
        place_id: UUID of the restaurant
        dish_name: Name of the dish
        db: Database session

    Returns:
        Deletion confirmation
    """
    try:
        logger.info(f"Deleting taste card for '{dish_name}' at place {place_id}")

        taste_card_service = TasteCardService(db)

        # Delete taste card
        success = await taste_card_service.delete_taste_card(
            place_id=place_id,
            dish_name=dish_name
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Taste card for '{dish_name}' at place {place_id} not found"
            )

        return {
            "message": f"Taste card for '{dish_name}' deleted successfully",
            "place_id": str(place_id),
            "dish_name": dish_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting taste card: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete taste card: {str(e)}"
        )


@router.post("/batch-generate")
async def batch_generate_taste_cards(
    place_id: UUID,
    dish_names: List[str],
    force_regenerate: bool = False,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Generate taste cards for multiple dishes at once.

    Args:
        place_id: UUID of the restaurant
        dish_names: List of dish names to generate cards for
        force_regenerate: Whether to regenerate existing cards
        db: Database session

    Returns:
        Batch generation results
    """
    try:
        logger.info(f"Batch generating taste cards for {len(dish_names)} dishes at place {place_id}")

        taste_card_service = TasteCardService(db)
        place_service = PlaceService(db)

        # Verify place exists
        place = await place_service.get_place_by_id(place_id)
        if not place:
            raise HTTPException(
                status_code=404,
                detail=f"Place {place_id} not found"
            )

        # Generate taste cards
        results = await taste_card_service.batch_generate_taste_cards(
            place=place,
            dish_names=dish_names,
            force_regenerate=force_regenerate
        )

        return {
            "place_id": str(place_id),
            "total_requested": len(dish_names),
            "generated": len(results["generated"]),
            "skipped": len(results["skipped"]),
            "failed": len(results["failed"]),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch taste card generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to batch generate taste cards: {str(e)}"
        )