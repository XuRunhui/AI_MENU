"""Places (restaurant) endpoints."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from uuid import UUID

from app.db.base import get_db
from app.schemas.place import PlaceLookupRequest, PlaceLookupResponse, PlaceResponse
from app.services.place_service import PlaceService
from app.services.review_service import ReviewService

router = APIRouter()


@router.post("/lookup", response_model=PlaceLookupResponse)
async def lookup_place(
    place_request: PlaceLookupRequest,
    db: AsyncSession = Depends(get_db)
) -> PlaceLookupResponse:
    """
    Look up a restaurant and fetch its reviews for taste analysis.

    This endpoint:
    1. Searches for the restaurant using Google Places/Yelp APIs
    2. Fetches recent reviews from multiple sources
    3. Caches restaurant data and reviews for 24 hours
    4. Returns restaurant info with review fetch statistics

    Args:
        place_request: Restaurant lookup details (name, city, address)
        db: Database session

    Returns:
        PlaceLookupResponse with restaurant data and review stats

    Example:
        ```python
        request_data = {
            "restaurant_name": "Szechuan Chef",
            "city": "Seattle",
            "address": "123 Pine St"
        }
        response = requests.post("/api/v1/places/lookup", json=request_data)

        # Response includes restaurant data and cached reviews
        {
            "place": {
                "id": "uuid-123",
                "name": "Szechuan Chef",
                "city": "Seattle",
                "rating": 4.3,
                "review_count": 205,
                "cuisine_tags": ["chinese", "sichuan"]
            },
            "reviews_fetched": 120,
            "cache_hit": false,
            "processing_stats": {
                "google_reviews": 120,
                "yelp_reviews": 85,
                "processing_time_ms": 890
            }
        }
        ```
    """
    try:
        logger.info(f"Looking up place: {place_request.restaurant_name} in {place_request.city}")

        # Initialize services
        place_service = PlaceService(db)
        review_service = ReviewService(db)

        # Look up or create place
        place, cache_hit = await place_service.lookup_place(
            name=place_request.restaurant_name,
            city=place_request.city,
            address=place_request.address,
            lat=place_request.lat,
            lng=place_request.lng
        )

        logger.info(f"Place lookup result: cache_hit={cache_hit}, place_id={place.id}")

        # Fetch reviews if not cached or cache expired
        reviews_fetched = 0
        processing_stats = {}

        if not cache_hit or not place.is_cache_valid:
            logger.info(f"Fetching fresh reviews for place {place.id}")
            reviews_fetched, processing_stats = await review_service.fetch_and_store_reviews(place)

        return PlaceLookupResponse(
            place=PlaceResponse.from_orm(place),
            reviews_fetched=reviews_fetched,
            cache_hit=cache_hit,
            processing_stats=processing_stats
        )

    except Exception as e:
        logger.error(f"Error looking up place: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to lookup place: {str(e)}"
        )


@router.get("/{place_id}", response_model=PlaceResponse)
async def get_place(
    place_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> PlaceResponse:
    """
    Get detailed information about a specific place.

    Args:
        place_id: UUID of the place
        db: Database session

    Returns:
        PlaceResponse with detailed place information
    """
    try:
        place_service = PlaceService(db)
        place = await place_service.get_place_by_id(place_id)

        if not place:
            raise HTTPException(
                status_code=404,
                detail=f"Place {place_id} not found"
            )

        return PlaceResponse.from_orm(place)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting place {place_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get place: {str(e)}"
        )


@router.post("/{place_id}/refresh-cache")
async def refresh_place_cache(
    place_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Force refresh cached data for a place.

    This endpoint:
    1. Re-fetches restaurant data from external APIs
    2. Updates cached reviews and ratings
    3. Resets cache expiration timer

    Args:
        place_id: UUID of the place to refresh
        db: Database session

    Returns:
        Refresh operation results
    """
    try:
        logger.info(f"Refreshing cache for place {place_id}")

        place_service = PlaceService(db)
        review_service = ReviewService(db)

        # Get place
        place = await place_service.get_place_by_id(place_id)
        if not place:
            raise HTTPException(
                status_code=404,
                detail=f"Place {place_id} not found"
            )

        # Force refresh reviews
        reviews_fetched, processing_stats = await review_service.fetch_and_store_reviews(
            place, force_refresh=True
        )

        # Update place metadata
        await place_service.update_cache_metadata(place)

        return {
            "place_id": str(place_id),
            "reviews_fetched": reviews_fetched,
            "cache_refreshed_at": place.cache_expires_at.isoformat() if place.cache_expires_at else None,
            "processing_stats": processing_stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing cache for place {place_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh cache: {str(e)}"
        )


@router.get("/search")
async def search_places(
    query: str = Query(..., description="Search query for restaurant names"),
    city: Optional[str] = Query(None, description="Filter by city"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Search for places by name and location.

    Args:
        query: Restaurant name search query
        city: Optional city filter
        limit: Maximum number of results
        db: Database session

    Returns:
        List of matching places
    """
    try:
        place_service = PlaceService(db)
        places = await place_service.search_places(
            query=query,
            city=city,
            limit=limit
        )

        return {
            "places": [PlaceResponse.from_orm(place) for place in places],
            "total": len(places),
            "query": query,
            "city_filter": city
        }

    except Exception as e:
        logger.error(f"Error searching places: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search places: {str(e)}"
        )