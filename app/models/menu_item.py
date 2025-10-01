"""Menu item model."""

from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class MenuItem(Base):
    """Menu item/dish model."""

    __tablename__ = "menu_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    place_id = Column(UUID(as_uuid=True), ForeignKey("places.id"), nullable=False, index=True)

    # Basic dish info
    name = Column(String(255), nullable=False, index=True)
    normalized_name = Column(String(255), index=True)  # Lowercase, cleaned for matching
    aliases = Column(JSON, default=[])  # Alternative names found in reviews
    description = Column(Text)  # Original menu description if available

    # Menu location
    section = Column(String(100))  # "Appetizers", "Main Dishes", etc.
    menu_position = Column(Integer)  # Order on menu

    # Pricing
    price_text = Column(String(50))  # Original price string from OCR
    price_amount = Column(Float)  # Parsed numeric price
    price_currency = Column(String(3), default="USD")

    # OCR metadata
    bbox = Column(JSON)  # Bounding box coordinates from OCR
    confidence_score = Column(Float)  # OCR confidence

    # Review matching stats
    review_mentions = Column(Integer, default=0)  # How many reviews mention this dish
    sentiment_score = Column(Float)  # Average sentiment from matched reviews

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    place = relationship("Place", back_populates="menu_items")
    taste_card = relationship("TasteCard", back_populates="menu_item", uselist=False)

    def __repr__(self) -> str:
        return f"<MenuItem(name='{self.name}', place_id='{self.place_id}')>"

    @property
    def display_price(self) -> str:
        """Get formatted price string."""
        if self.price_amount:
            return f"${self.price_amount:.2f}"
        return self.price_text or "Price not available"

    @property
    def search_terms(self) -> list[str]:
        """Get all terms to use when searching reviews."""
        terms = [self.name, self.normalized_name]
        if self.aliases:
            terms.extend(self.aliases)
        return [t for t in terms if t]  # Remove empty strings