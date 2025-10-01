"""Combo recommendation model."""

from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class ComboTemplate(Base):
    """Saved dish combination recommendations."""

    __tablename__ = "combo_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    place_id = Column(UUID(as_uuid=True), ForeignKey("places.id"), nullable=False, index=True)

    # Combo constraints
    party_size_min = Column(Integer, nullable=False)
    party_size_max = Column(Integer, nullable=False)
    budget_min = Column(Float)
    budget_max = Column(Float)

    # Combo items (stored as menu item IDs)
    menu_item_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False)
    item_names = Column(ARRAY(String(255)), nullable=False)  # Cached for quick display

    # Scoring and rationale
    combo_score = Column(Float, nullable=False)  # Algorithm-generated score
    rationale = Column(Text)  # Human-readable explanation
    balance_matrix = Column(JSON, default={})  # Detailed balance breakdown
    """
    Example balance_matrix:
    {
        "spice_range": [0, 3],                     # Min and max spice levels
        "texture_variety": ["silky", "crispy", "fluffy"],
        "richness_variety": ["heavy", "light", "light"],
        "protein_coverage": ["pork", "vegetable"],
        "cooking_methods": ["braised", "stir-fried", "steamed"]
    }
    """

    # Constraint compatibility
    dietary_tags = Column(ARRAY(String(50)), default=[])  # "vegetarian", "gluten-free", etc.
    allergen_warnings = Column(ARRAY(String(50)), default=[])  # "contains peanuts"
    spice_level_max = Column(Integer, default=3)  # Maximum spice in combo

    # Usage statistics
    suggestion_count = Column(Integer, default=0)  # How many times suggested
    acceptance_count = Column(Integer, default=0)  # How many times accepted
    last_suggested = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Hash for constraint matching
    constraints_hash = Column(String(64), index=True)  # MD5 of constraints for quick lookup

    # Relationships
    place = relationship("Place", back_populates="combo_templates")

    def __repr__(self) -> str:
        return f"<ComboTemplate(place_id='{self.place_id}', party_size={self.party_size_min}-{self.party_size_max})>"

    @property
    def estimated_total_price(self) -> Optional[float]:
        """Calculate estimated total price if item prices are available."""
        # This would require joining with menu_items to get actual prices
        # Implementation would be in a service layer
        return None

    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate percentage."""
        if self.suggestion_count == 0:
            return 0.0
        return (self.acceptance_count / self.suggestion_count) * 100

    @property
    def is_suitable_for_party_size(self, party_size: int) -> bool:
        """Check if combo is suitable for given party size."""
        return self.party_size_min <= party_size <= self.party_size_max

    @property
    def spice_level_description(self) -> str:
        """Get human-readable spice level."""
        levels = {0: "No spice", 1: "Mild", 2: "Medium", 3: "Very spicy"}
        return levels.get(self.spice_level_max, "Unknown")

    def matches_dietary_constraints(self, constraints: List[str]) -> bool:
        """Check if combo is compatible with dietary constraints."""
        # If user requires vegetarian, combo must be tagged vegetarian
        for constraint in constraints:
            if constraint not in self.dietary_tags:
                return False
        return True

    def has_allergen_conflicts(self, allergies: List[str]) -> bool:
        """Check if combo contains allergens user is allergic to."""
        return any(allergen in self.allergen_warnings for allergen in allergies)

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "id": str(self.id),
            "items": self.item_names,
            "rationale": self.rationale,
            "score": self.combo_score,
            "party_size_range": [self.party_size_min, self.party_size_max],
            "budget_range": [self.budget_min, self.budget_max] if self.budget_min else None,
            "balance_matrix": self.balance_matrix,
            "dietary_tags": self.dietary_tags,
            "allergen_warnings": self.allergen_warnings,
            "spice_level": self.spice_level_description,
            "acceptance_rate": self.acceptance_rate,
        }