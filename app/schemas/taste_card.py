"""Taste card related schemas."""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from uuid import UUID
from datetime import datetime


class TasteAspects(BaseModel):
    """Schema for taste aspects extracted from reviews."""
    spice: int = Field(0, ge=0, le=3, description="Spice level (0=none, 3=very spicy)")
    salt: int = Field(0, ge=0, le=3, description="Saltiness level")
    sweetness: int = Field(0, ge=0, le=3, description="Sweetness level")
    sour: int = Field(0, ge=0, le=3, description="Sourness level")
    umami: int = Field(0, ge=0, le=3, description="Umami level")

    heat_type: List[str] = Field(default_factory=list, description="Type of heat (chili, peppercorn, wasabi)")
    texture: List[str] = Field(default_factory=list, description="Texture descriptors")
    richness: Optional[str] = Field(None, description="Richness level (light, medium, heavy)")
    portion: Optional[str] = Field(None, description="Portion size (small, medium, large)")

    allergens: List[str] = Field(default_factory=list, description="Common allergens")
    tags: List[str] = Field(default_factory=list, description="Additional descriptive tags")

    @validator("heat_type", "texture", "allergens", "tags", pre=True)
    def ensure_list(cls, v):
        """Ensure fields are lists."""
        if v is None:
            return []
        if not isinstance(v, list):
            return [v] if v else []
        return v

    @validator("richness")
    def validate_richness(cls, v):
        """Validate richness level."""
        if v is not None and v not in ["light", "medium", "heavy"]:
            raise ValueError("Richness must be 'light', 'medium', or 'heavy'")
        return v

    @validator("portion")
    def validate_portion(cls, v):
        """Validate portion size."""
        if v is not None and v not in ["small", "medium", "large"]:
            raise ValueError("Portion must be 'small', 'medium', or 'large'")
        return v


class TasteCardRequest(BaseModel):
    """Request schema for generating taste cards."""
    place_id: UUID = Field(..., description="Restaurant place ID")
    dish_name: str = Field(..., description="Name of the dish")
    force_regenerate: bool = Field(False, description="Force regeneration even if card exists")

    @validator("dish_name")
    def validate_dish_name(cls, v):
        """Ensure dish name is not empty."""
        if not v or not v.strip():
            raise ValueError("Dish name cannot be empty")
        return v.strip()


class TasteCardResponse(BaseModel):
    """Response schema for taste cards."""
    id: Optional[UUID] = Field(None, description="Taste card ID (if saved)")
    dish_name: str = Field(..., description="Name of the dish")
    restaurant_name: Optional[str] = Field(None, description="Restaurant name")

    # Generated content
    bullets: List[str] = Field(..., description="Taste description bullets")
    pairing_suggestion: Optional[str] = Field(None, description="Pairing recommendations")
    serving_notes: Optional[str] = Field(None, description="Serving size/sharing notes")

    # Taste data
    aspects: TasteAspects = Field(..., description="Structured taste attributes")

    # Metadata
    confidence_level: str = Field(..., description="Confidence level (low, medium, high)")
    fallback_level: str = Field(..., description="Data source level (A, B, C, D)")
    sources: List[str] = Field(..., description="Source descriptions")
    review_snippets_used: int = Field(0, description="Number of review snippets analyzed")

    # Timestamps
    generated_at: Optional[datetime] = Field(None, description="When taste card was generated")

    @validator("confidence_level")
    def validate_confidence(cls, v):
        """Validate confidence level."""
        if v not in ["low", "medium", "high"]:
            raise ValueError("Confidence level must be 'low', 'medium', or 'high'")
        return v

    @validator("fallback_level")
    def validate_fallback_level(cls, v):
        """Validate fallback level."""
        if v not in ["A", "B", "C", "D"]:
            raise ValueError("Fallback level must be 'A', 'B', 'C', or 'D'")
        return v

    @property
    def spice_description(self) -> str:
        """Get human-readable spice level."""
        spice_map = {0: "No spice", 1: "Mild", 2: "Medium", 3: "Very spicy"}
        return spice_map.get(self.aspects.spice, "Unknown")

    @property
    def has_allergens(self) -> bool:
        """Check if dish contains allergens."""
        return len(self.aspects.allergens) > 0

    class Config:
        from_attributes = True