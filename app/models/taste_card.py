"""Taste card model for dish descriptions."""

from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class TasteCard(Base):
    """Generated taste description for a menu item."""

    __tablename__ = "taste_cards"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    menu_item_id = Column(UUID(as_uuid=True), ForeignKey("menu_items.id"), nullable=False, unique=True, index=True)
    place_id = Column(UUID(as_uuid=True), ForeignKey("places.id"), nullable=False, index=True)

    # Taste attributes (extracted from reviews)
    aspects = Column(JSON, nullable=False, default={})  # Structured taste data
    """
    Example aspects JSON:
    {
        "spice": 3,                           # 0-3 scale
        "salt": 2,
        "sweetness": 1,
        "sour": 0,
        "umami": 3,
        "heat_type": ["peppercorn", "chili"],
        "texture": ["silky", "soft"],
        "richness": "heavy",                  # "light", "medium", "heavy"
        "portion": "large",                   # "small", "medium", "large"
        "allergens": ["peanut"],
        "tags": ["authentic", "numbing", "oily"]
    }
    """

    # Generated descriptions
    bullets = Column(ARRAY(Text), nullable=False, default=[])  # Taste description bullets
    pairing_suggestion = Column(Text)  # "Pairs well with..."
    serving_notes = Column(Text)  # "Best shared between 2 people"

    # Generation metadata
    confidence_level = Column(String(20), nullable=False)  # "low", "medium", "high"
    fallback_level = Column(String(1), nullable=False)  # "A", "B", "C", "D"
    """
    Fallback levels:
    A: Restaurant-specific reviews
    B: City-wide dish reviews
    C: Global dish knowledge
    D: LLM prior knowledge only
    """

    # Source attribution
    sources = Column(JSON, default=[])  # Source descriptions for transparency
    """
    Example sources:
    ["120+ Google reviews", "85 Yelp reviews", "Top 3 photos show authentic presentation"]
    """

    review_snippets_used = Column(Integer, default=0)  # Number of review snippets analyzed
    model_version = Column(String(50))  # LLM model used for generation

    # Timestamps
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    menu_item = relationship("MenuItem", back_populates="taste_card")
    place = relationship("Place", back_populates="taste_cards")

    def __repr__(self) -> str:
        return f"<TasteCard(menu_item_id='{self.menu_item_id}', confidence='{self.confidence_level}')>"

    @property
    def spice_level(self) -> str:
        """Get human-readable spice level."""
        spice_score = self.aspects.get("spice", 0)
        levels = {0: "None", 1: "Mild", 2: "Medium", 3: "Very Spicy"}
        return levels.get(spice_score, "Unknown")

    @property
    def has_allergens(self) -> bool:
        """Check if dish contains common allergens."""
        allergens = self.aspects.get("allergens", [])
        return len(allergens) > 0

    @property
    def texture_summary(self) -> str:
        """Get comma-separated texture description."""
        textures = self.aspects.get("texture", [])
        return ", ".join(textures) if textures else "Not specified"

    @property
    def confidence_score(self) -> float:
        """Get numeric confidence score (0.0-1.0)."""
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        return confidence_map.get(self.confidence_level, 0.0)

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "dish_name": self.menu_item.name if self.menu_item else "Unknown",
            "bullets": self.bullets,
            "aspects": self.aspects,
            "pairing": self.pairing_suggestion,
            "serving_notes": self.serving_notes,
            "confidence": self.confidence_level,
            "sources": self.sources,
            "fallback_level": self.fallback_level,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }