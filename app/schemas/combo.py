"""Combo recommendation related schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID


class ComboRequest(BaseModel):
    """Request schema for combo recommendations."""
    place_id: UUID = Field(..., description="Restaurant place ID")
    party_size: int = Field(..., ge=1, le=10, description="Number of people (1-10)")
    budget: Optional[float] = Field(None, ge=0, description="Budget limit in USD")

    # Dietary constraints
    dietary_constraints: List[str] = Field(
        default_factory=list,
        description="Dietary requirements (vegetarian, vegan, gluten-free, etc.)"
    )
    allergies: List[str] = Field(
        default_factory=list,
        description="Food allergies (peanut, shellfish, dairy, etc.)"
    )

    # Preferences
    spice_preference: int = Field(2, ge=0, le=3, description="Max spice level (0-3)")
    cuisine_preference: Optional[str] = Field(None, description="Preferred cuisine style")

    # Options
    max_combos: int = Field(3, ge=1, le=5, description="Maximum number of combos to return")
    include_alternatives: bool = Field(True, description="Include alternative item suggestions")

    @validator("dietary_constraints", "allergies", pre=True)
    def ensure_list(cls, v):
        """Ensure fields are lists."""
        if v is None:
            return []
        if not isinstance(v, list):
            return [v] if v else []
        return [item.lower().strip() for item in v if item]


class ComboItem(BaseModel):
    """Individual item in a combo recommendation."""
    menu_item_id: UUID = Field(..., description="Menu item ID")
    name: str = Field(..., description="Dish name")
    price: Optional[float] = Field(None, description="Item price")
    role: str = Field(..., description="Role in combo (protein, vegetable, carb, etc.)")

    # Taste summary
    spice_level: int = Field(0, ge=0, le=3, description="Spice level")
    main_flavors: List[str] = Field(default_factory=list, description="Primary flavors")
    texture: List[str] = Field(default_factory=list, description="Texture descriptors")

    class Config:
        from_attributes = True


class BalanceMatrix(BaseModel):
    """Balance analysis for a combo."""
    spice_range: List[int] = Field(..., description="Min and max spice levels in combo")
    texture_variety: List[str] = Field(..., description="Different textures represented")
    richness_variety: List[str] = Field(..., description="Different richness levels")
    protein_coverage: List[str] = Field(default_factory=list, description="Protein types")
    cooking_methods: List[str] = Field(default_factory=list, description="Cooking techniques")
    flavor_balance: Dict[str, int] = Field(default_factory=dict, description="Flavor dimension scores")


class ComboRecommendation(BaseModel):
    """Single combo recommendation."""
    id: Optional[UUID] = Field(None, description="Combo template ID (if saved)")
    items: List[ComboItem] = Field(..., description="Items in this combo")

    # Scoring and rationale
    score: float = Field(..., description="Algorithm-generated score")
    rationale: str = Field(..., description="Human-readable explanation")
    balance_matrix: BalanceMatrix = Field(..., description="Detailed balance analysis")

    # Practical info
    estimated_total: Optional[float] = Field(None, description="Estimated total price")
    estimated_portions: str = Field(..., description="Portion guidance")
    serving_order: Optional[List[str]] = Field(None, description="Suggested serving order")

    # Compatibility
    suitable_for_party_size: bool = Field(..., description="Suitable for requested party size")
    meets_dietary_requirements: bool = Field(..., description="Meets all dietary constraints")
    allergen_warnings: List[str] = Field(default_factory=list, description="Allergen warnings")

    # Alternatives
    alternatives: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Alternative items for each role"
    )

    @property
    def max_spice_level(self) -> int:
        """Get maximum spice level in combo."""
        return max((item.spice_level for item in self.items), default=0)

    @property
    def item_count(self) -> int:
        """Get number of items in combo."""
        return len(self.items)


class ComboResponse(BaseModel):
    """Response schema for combo recommendations."""
    combos: List[ComboRecommendation] = Field(..., description="Recommended combos")

    # Request context
    place_name: str = Field(..., description="Restaurant name")
    party_size: int = Field(..., description="Requested party size")
    budget_requested: Optional[float] = Field(None, description="Requested budget")
    constraints_applied: Dict[str, Any] = Field(..., description="Applied constraints summary")

    # Processing metadata
    total_items_available: int = Field(..., description="Total menu items considered")
    combinations_evaluated: int = Field(..., description="Number of combinations scored")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

    @validator("combos")
    def validate_combos(cls, v):
        """Ensure at least one combo is returned."""
        if not v:
            raise ValueError("No suitable combos found")
        return v

    @property
    def budget_range(self) -> Optional[Dict[str, float]]:
        """Get price range of recommended combos."""
        if not self.combos:
            return None

        prices = [c.estimated_total for c in self.combos if c.estimated_total]
        if not prices:
            return None

        return {
            "min": min(prices),
            "max": max(prices),
            "average": sum(prices) / len(prices)
        }