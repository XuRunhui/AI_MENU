"""Place (restaurant) model."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class Place(Base):
    """Restaurant/place model."""

    __tablename__ = "places"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    address = Column(Text)
    city = Column(String(100), index=True)
    lat = Column(Float)
    lng = Column(Float)

    # External service IDs
    google_place_id = Column(String(255), unique=True, index=True)
    yelp_business_id = Column(String(255), unique=True, index=True)

    # Basic info
    phone = Column(String(50))
    website = Column(String(500))
    rating = Column(Float)  # Average rating
    review_count = Column(Integer, default=0)
    price_level = Column(Integer)  # 1-4 scale

    # Categorization
    cuisine_tags = Column(ARRAY(String(50)), default=[])  # ["chinese", "sichuan"]
    category = Column(String(100))  # "Restaurant", "Fast Food", etc.

    # Cache metadata
    reviews_last_fetched = Column(DateTime(timezone=True))
    cache_expires_at = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    menu_items = relationship("MenuItem", back_populates="place", cascade="all, delete-orphan")
    reviews = relationship("Review", back_populates="place", cascade="all, delete-orphan")
    taste_cards = relationship("TasteCard", back_populates="place", cascade="all, delete-orphan")
    combo_templates = relationship("ComboTemplate", back_populates="place", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Place(name='{self.name}', city='{self.city}')>"

    @property
    def is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if not self.cache_expires_at:
            return False
        return datetime.utcnow() < self.cache_expires_at

    @property
    def display_location(self) -> str:
        """Get formatted location string."""
        if self.city and self.address:
            return f"{self.address}, {self.city}"
        return self.city or self.address or "Unknown location"