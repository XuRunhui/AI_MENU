"""Place (restaurant) related schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID
from datetime import datetime


class PlaceLookupRequest(BaseModel):
    """Request schema for restaurant lookup."""
    restaurant_name: str = Field(..., description="Restaurant name")
    city: Optional[str] = Field(None, description="City location")
    address: Optional[str] = Field(None, description="Full address")
    lat: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    lng: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")

    @validator("restaurant_name")
    def validate_restaurant_name(cls, v):
        """Ensure restaurant name is not empty."""
        if not v or not v.strip():
            raise ValueError("Restaurant name cannot be empty")
        return v.strip()


class PlaceResponse(BaseModel):
    """Response schema for restaurant/place data."""
    id: UUID
    name: str
    address: Optional[str]
    city: Optional[str]
    display_location: str
    lat: Optional[float]
    lng: Optional[float]

    # External IDs
    google_place_id: Optional[str]
    yelp_business_id: Optional[str]

    # Basic info
    phone: Optional[str]
    website: Optional[str]
    rating: Optional[float]
    review_count: int
    price_level: Optional[int]

    # Categorization
    cuisine_tags: List[str]
    category: Optional[str]

    # Cache status
    is_cache_valid: bool
    reviews_last_fetched: Optional[datetime]

    class Config:
        from_attributes = True


class PlaceLookupResponse(BaseModel):
    """Response schema for place lookup with review fetch status."""
    place: PlaceResponse
    reviews_fetched: int = Field(..., description="Number of reviews fetched")
    cache_hit: bool = Field(..., description="Whether data was served from cache")
    processing_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processing statistics"
    )
    """
    Example processing_stats:
    {
        "api_calls_made": 2,
        "google_reviews": 120,
        "yelp_reviews": 85,
        "processing_time_ms": 890,
        "review_languages": ["en", "es"],
        "average_rating": 4.3
    }
    """