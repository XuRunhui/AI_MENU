"""Menu service for processing extracted menu items."""

import re
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.models.menu_item import MenuItem
from app.models.place import Place
from app.schemas.menu import MenuItemCreate, MenuItemResponse


class MenuService:
    """Service for processing and managing menu items."""

    def __init__(self, db: AsyncSession):
        """
        Initialize menu service.

        Args:
            db: Database session
        """
        self.db = db

    async def process_extracted_items(
        self,
        extracted_items: List[MenuItemCreate],
        restaurant_name: Optional[str] = None,
        city: Optional[str] = None
    ) -> List[MenuItemResponse]:
        """
        Process extracted menu items and optionally save to database.

        Args:
            extracted_items: List of items from OCR
            restaurant_name: Optional restaurant name for context
            city: Optional city for context

        Returns:
            List of processed menu item responses
        """
        logger.info(f"Processing {len(extracted_items)} extracted menu items")

        processed_items = []

        for item in extracted_items:
            try:
                # Normalize the item
                normalized_item = self._normalize_menu_item(item)

                # Convert to response format (without saving to DB yet)
                response_item = MenuItemResponse(
                    id=UUID('00000000-0000-0000-0000-000000000000'),  # Placeholder
                    name=normalized_item.name,
                    normalized_name=self._normalize_dish_name(normalized_item.name),
                    section=normalized_item.section,
                    price_text=normalized_item.price_text,
                    price_amount=normalized_item.price_amount,
                    display_price=self._format_display_price(
                        normalized_item.price_amount,
                        normalized_item.price_text
                    ),
                    bbox=normalized_item.bbox.dict() if normalized_item.bbox else None,
                    confidence_score=normalized_item.confidence_score,
                    review_mentions=0,  # Will be populated later
                    sentiment_score=None  # Will be populated later
                )

                processed_items.append(response_item)

            except Exception as e:
                logger.error(f"Failed to process menu item '{item.name}': {str(e)}")
                continue

        logger.info(f"Successfully processed {len(processed_items)} menu items")
        return processed_items

    def _normalize_menu_item(self, item: MenuItemCreate) -> MenuItemCreate:
        """
        Normalize a menu item for consistency.

        Args:
            item: Raw menu item from OCR

        Returns:
            Normalized menu item
        """
        # Normalize dish name
        normalized_name = self._normalize_dish_name(item.name)

        # Normalize section
        normalized_section = self._normalize_section(item.section)

        # Create normalized item
        return MenuItemCreate(
            name=normalized_name,
            section=normalized_section,
            price_text=item.price_text,
            price_amount=item.price_amount,
            bbox=item.bbox,
            confidence_score=item.confidence_score
        )

    def _normalize_dish_name(self, name: str) -> str:
        """
        Normalize a dish name for consistency and searchability.

        Args:
            name: Raw dish name

        Returns:
            Normalized dish name
        """
        if not name:
            return ""

        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name.strip())

        # Remove trailing dots, dashes, or underscores (menu formatting)
        name = re.sub(r'[\.\_\-]+$', '', name)

        # Remove price information if it got mixed in
        name = re.sub(r'\$\d+(\.\d{2})?', '', name)

        # Remove common OCR artifacts
        name = re.sub(r'[^\w\s\-\'\(\)\&]', '', name)

        # Normalize common abbreviations
        abbreviations = {
            r'\bw/\b': 'with',
            r'\b&\b': 'and',
            r'\bchk\b': 'chicken',
            r'\bveg\b': 'vegetable',
            r'\bfr\b': 'fried',
            r'\bst\b': 'stir',
        }

        name_lower = name.lower()
        for pattern, replacement in abbreviations.items():
            name_lower = re.sub(pattern, replacement, name_lower)

        # Convert back to title case
        name = name_lower.title()

        # Fix common casing issues
        name = re.sub(r'\bAnd\b', 'and', name)
        name = re.sub(r'\bWith\b', 'with', name)
        name = re.sub(r'\bIn\b', 'in', name)
        name = re.sub(r'\bOf\b', 'of', name)

        return name.strip()

    def _normalize_section(self, section: Optional[str]) -> Optional[str]:
        """
        Normalize menu section names.

        Args:
            section: Raw section name

        Returns:
            Normalized section name
        """
        if not section:
            return None

        section = section.strip().title()

        # Normalize common section variations
        section_mappings = {
            'Appetizer': 'Appetizers',
            'Starter': 'Appetizers',
            'App': 'Appetizers',
            'Main Course': 'Main Dishes',
            'Main': 'Main Dishes',
            'Entree': 'Main Dishes',
            'Entrees': 'Main Dishes',
            'Entre': 'Main Dishes',
            'Mains': 'Main Dishes',
            'Specialty': 'Specialties',
            'Special': 'Specialties',
            'Dessert': 'Desserts',
            'Sweet': 'Desserts',
            'Beverage': 'Beverages',
            'Drink': 'Beverages',
            'Drinks': 'Beverages',
            'Side': 'Sides',
            'Side Dish': 'Sides',
        }

        return section_mappings.get(section, section)

    def _format_display_price(
        self,
        price_amount: Optional[float],
        price_text: Optional[str]
    ) -> str:
        """
        Format price for display.

        Args:
            price_amount: Parsed numeric price
            price_text: Original price text

        Returns:
            Formatted price string
        """
        if price_amount:
            return f"${price_amount:.2f}"
        elif price_text:
            return price_text
        else:
            return "Price not available"

    async def save_menu_items(
        self,
        items: List[MenuItemCreate],
        place_id: UUID
    ) -> List[MenuItem]:
        """
        Save menu items to database for a specific place.

        Args:
            items: List of menu items to save
            place_id: UUID of the restaurant

        Returns:
            List of saved MenuItem objects
        """
        logger.info(f"Saving {len(items)} menu items for place {place_id}")

        saved_items = []

        for item in items:
            try:
                # Check if item already exists (avoid duplicates)
                existing = await self._find_existing_menu_item(item.name, place_id)

                if existing:
                    logger.debug(f"Menu item '{item.name}' already exists, skipping")
                    saved_items.append(existing)
                    continue

                # Create new menu item
                menu_item = MenuItem(
                    place_id=place_id,
                    name=item.name,
                    normalized_name=self._normalize_dish_name(item.name),
                    section=item.section,
                    price_text=item.price_text,
                    price_amount=item.price_amount,
                    bbox=item.bbox.dict() if item.bbox else None,
                    confidence_score=item.confidence_score
                )

                self.db.add(menu_item)
                saved_items.append(menu_item)

            except Exception as e:
                logger.error(f"Failed to save menu item '{item.name}': {str(e)}")
                continue

        # Commit all items
        await self.db.commit()

        # Refresh objects to get IDs
        for item in saved_items:
            await self.db.refresh(item)

        logger.info(f"Successfully saved {len(saved_items)} menu items")
        return saved_items

    async def _find_existing_menu_item(
        self,
        dish_name: str,
        place_id: UUID
    ) -> Optional[MenuItem]:
        """
        Find existing menu item by name and place.

        Args:
            dish_name: Name of the dish
            place_id: UUID of the place

        Returns:
            Existing MenuItem if found
        """
        normalized_name = self._normalize_dish_name(dish_name)

        query = select(MenuItem).where(
            and_(
                MenuItem.place_id == place_id,
                MenuItem.normalized_name == normalized_name
            )
        )

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_menu_items_for_place(
        self,
        place_id: UUID,
        section: Optional[str] = None
    ) -> List[MenuItem]:
        """
        Get all menu items for a place.

        Args:
            place_id: UUID of the place
            section: Optional section filter

        Returns:
            List of menu items
        """
        query = select(MenuItem).where(MenuItem.place_id == place_id)

        if section:
            query = query.where(MenuItem.section == section)

        query = query.order_by(MenuItem.section, MenuItem.name)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def search_menu_items(
        self,
        place_id: UUID,
        search_term: str,
        limit: int = 20
    ) -> List[MenuItem]:
        """
        Search menu items by name.

        Args:
            place_id: UUID of the place
            search_term: Search term
            limit: Maximum results

        Returns:
            List of matching menu items
        """
        normalized_term = self._normalize_dish_name(search_term)

        query = select(MenuItem).where(
            and_(
                MenuItem.place_id == place_id,
                MenuItem.normalized_name.ilike(f"%{normalized_term}%")
            )
        ).limit(limit)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def update_menu_item_stats(
        self,
        menu_item_id: UUID,
        review_mentions: int,
        sentiment_score: Optional[float] = None
    ) -> None:
        """
        Update menu item statistics from review analysis.

        Args:
            menu_item_id: UUID of the menu item
            review_mentions: Number of review mentions found
            sentiment_score: Average sentiment score
        """
        query = select(MenuItem).where(MenuItem.id == menu_item_id)
        result = await self.db.execute(query)
        menu_item = result.scalar_one_or_none()

        if menu_item:
            menu_item.review_mentions = review_mentions
            if sentiment_score is not None:
                menu_item.sentiment_score = sentiment_score

            await self.db.commit()