"""Combo recommendation endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from uuid import UUID

from app.db.base import get_db
from app.schemas.combo import ComboRequest, ComboResponse
from app.services.combo_service import ComboService
from app.services.place_service import PlaceService

router = APIRouter()


@router.post("/recommend", response_model=ComboResponse)
async def recommend_combos(
    combo_request: ComboRequest,
    db: AsyncSession = Depends(get_db)
) -> ComboResponse:
    """
    Generate balanced dish combination recommendations for a party.

    This endpoint:
    1. Analyzes all available dishes at the restaurant
    2. Scores combinations based on balance (spice, texture, richness)
    3. Applies dietary constraints and budget limits
    4. Returns top-scored combos with rationale and alternatives

    Args:
        combo_request: Party details and preferences
        db: Database session

    Returns:
        ComboResponse with recommended dish combinations

    Example:
        ```python
        request_data = {
            "place_id": "uuid-123",
            "party_size": 3,
            "budget": 75.0,
            "dietary_constraints": ["vegetarian"],
            "allergies": ["peanut"],
            "spice_preference": 2,
            "max_combos": 3
        }
        response = requests.post("/api/v1/combos/recommend", json=request_data)

        # Response includes balanced combos with rationale
        {
            "combos": [
                {
                    "items": [
                        {"name": "Mapo Tofu", "role": "protein", "spice_level": 3},
                        {"name": "Dry-Fried Green Beans", "role": "vegetable", "spice_level": 1},
                        {"name": "Steamed Rice", "role": "carb", "spice_level": 0}
                    ],
                    "score": 87.5,
                    "rationale": "Balanced heat and texture - spicy tofu with mild crispy beans",
                    "estimated_total": 42.50,
                    "meets_dietary_requirements": true
                }
            ],
            "place_name": "Szechuan Chef",
            "party_size": 3,
            "total_items_available": 45,
            "combinations_evaluated": 1200
        }
        ```
    """
    try:
        logger.info(f"Generating combo recommendations for party of {combo_request.party_size} at place {combo_request.place_id}")

        # Initialize services
        combo_service = ComboService(db)
        place_service = PlaceService(db)

        # Verify place exists
        place = await place_service.get_place_by_id(combo_request.place_id)
        if not place:
            raise HTTPException(
                status_code=404,
                detail=f"Place {combo_request.place_id} not found"
            )

        # Check if place has sufficient menu data
        menu_items_count = await combo_service.get_menu_items_count(combo_request.place_id)
        if menu_items_count < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient menu data for combo recommendations. Found {menu_items_count} items, need at least 3."
            )

        # Generate combo recommendations
        combos = await combo_service.generate_combo_recommendations(
            place=place,
            party_size=combo_request.party_size,
            budget=combo_request.budget,
            dietary_constraints=combo_request.dietary_constraints,
            allergies=combo_request.allergies,
            spice_preference=combo_request.spice_preference,
            max_combos=combo_request.max_combos,
            include_alternatives=combo_request.include_alternatives
        )

        # Generate processing statistics
        processing_stats = await combo_service.get_processing_stats()

        return ComboResponse(
            combos=combos,
            place_name=place.name,
            party_size=combo_request.party_size,
            budget_requested=combo_request.budget,
            constraints_applied={
                "dietary_constraints": combo_request.dietary_constraints,
                "allergies": combo_request.allergies,
                "max_spice": combo_request.spice_preference,
                "budget": combo_request.budget
            },
            total_items_available=menu_items_count,
            combinations_evaluated=processing_stats["combinations_evaluated"],
            processing_time_ms=processing_stats["processing_time_ms"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating combo recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate combo recommendations: {str(e)}"
        )


@router.get("/{place_id}/saved", response_model=List[dict])
async def get_saved_combos(
    place_id: UUID,
    party_size: int = Query(None, ge=1, le=10, description="Filter by party size"),
    limit: int = Query(20, ge=1, le=50, description="Maximum combos to return"),
    db: AsyncSession = Depends(get_db)
) -> List[dict]:
    """
    Get previously saved combo templates for a restaurant.

    Args:
        place_id: UUID of the restaurant
        party_size: Optional party size filter
        limit: Maximum number of combos to return
        db: Database session

    Returns:
        List of saved combo templates
    """
    try:
        logger.info(f"Getting saved combos for place {place_id}")

        combo_service = ComboService(db)

        # Get saved combos
        saved_combos = await combo_service.get_saved_combos(
            place_id=place_id,
            party_size=party_size,
            limit=limit
        )

        return [combo.to_api_response() for combo in saved_combos]

    except Exception as e:
        logger.error(f"Error getting saved combos: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get saved combos: {str(e)}"
        )


@router.post("/{place_id}/feedback")
async def record_combo_feedback(
    place_id: UUID,
    combo_id: UUID,
    accepted: bool,
    feedback_notes: str = None,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Record user feedback on a combo recommendation.

    This helps improve future recommendations by tracking acceptance rates.

    Args:
        place_id: UUID of the restaurant
        combo_id: UUID of the combo recommendation
        accepted: Whether the user accepted the recommendation
        feedback_notes: Optional feedback text
        db: Database session

    Returns:
        Feedback recording confirmation
    """
    try:
        logger.info(f"Recording combo feedback: combo_id={combo_id}, accepted={accepted}")

        combo_service = ComboService(db)

        # Record feedback
        success = await combo_service.record_combo_feedback(
            combo_id=combo_id,
            accepted=accepted,
            feedback_notes=feedback_notes
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Combo {combo_id} not found"
            )

        return {
            "message": "Feedback recorded successfully",
            "combo_id": str(combo_id),
            "accepted": accepted,
            "feedback_notes": feedback_notes
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording combo feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record feedback: {str(e)}"
        )


@router.get("/{place_id}/analytics")
async def get_combo_analytics(
    place_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Get analytics data for combo recommendations at a restaurant.

    Args:
        place_id: UUID of the restaurant
        db: Database session

    Returns:
        Analytics data including acceptance rates and popular combos
    """
    try:
        logger.info(f"Getting combo analytics for place {place_id}")

        combo_service = ComboService(db)

        # Get analytics data
        analytics = await combo_service.get_combo_analytics(place_id)

        return {
            "place_id": str(place_id),
            "total_combos_suggested": analytics["total_suggested"],
            "total_combos_accepted": analytics["total_accepted"],
            "acceptance_rate": analytics["acceptance_rate"],
            "popular_combos": analytics["popular_combos"],
            "avg_party_size": analytics["avg_party_size"],
            "avg_budget": analytics["avg_budget"],
            "common_dietary_constraints": analytics["common_constraints"]
        }

    except Exception as e:
        logger.error(f"Error getting combo analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analytics: {str(e)}"
        )