"""Menu and OCR-related schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID


class BoundingBox(BaseModel):
    """Bounding box coordinates from OCR."""
    x: float = Field(..., description="X coordinate (left)")
    y: float = Field(..., description="Y coordinate (top)")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")


class OCRMenuRequest(BaseModel):
    """Request schema for OCR menu processing."""
    # In practice, this would be handled via file upload
    # The actual image data would be in the request files
    restaurant_name: Optional[str] = Field(None, description="Restaurant name (optional)")
    city: Optional[str] = Field(None, description="City location (optional)")


class MenuItemCreate(BaseModel):
    """Schema for creating a menu item from OCR."""
    name: str = Field(..., description="Dish name extracted from menu")
    section: Optional[str] = Field(None, description="Menu section (e.g., 'Appetizers')")
    price_text: Optional[str] = Field(None, description="Original price string from OCR")
    price_amount: Optional[float] = Field(None, description="Parsed numeric price")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box coordinates")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="OCR confidence")

    @validator("name")
    def validate_name(cls, v):
        """Ensure dish name is not empty."""
        if not v or not v.strip():
            raise ValueError("Dish name cannot be empty")
        return v.strip()


class MenuItemResponse(BaseModel):
    """Response schema for menu items."""
    id: UUID
    name: str
    normalized_name: Optional[str]
    section: Optional[str]
    price_text: Optional[str]
    price_amount: Optional[float]
    display_price: str
    bbox: Optional[Dict[str, Any]]
    confidence_score: Optional[float]
    review_mentions: int
    sentiment_score: Optional[float]

    class Config:
        from_attributes = True


class OCRMenuResponse(BaseModel):
    """Response schema for OCR menu processing."""
    menu_items: List[MenuItemResponse] = Field(..., description="Extracted menu items")
    processing_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="OCR processing statistics"
    )
    """
    Example processing_stats:
    {
        "total_items_detected": 25,
        "items_with_prices": 23,
        "average_confidence": 0.87,
        "processing_time_ms": 1250,
        "sections_detected": ["Appetizers", "Main Dishes", "Desserts"]
    }
    """

    @validator("menu_items")
    def validate_menu_items(cls, v):
        """Ensure at least one menu item was detected."""
        if not v:
            raise ValueError("No menu items detected")
        return v