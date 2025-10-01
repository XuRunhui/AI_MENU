"""Place (restaurant) service for lookup and management."""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
from uuid import UUID

import googlemaps
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.config import settings
from app.models.place import Place
from app.models.review import Review


class PlaceService:
    """Service for managing restaurant/place data."""

    def __init__(self, db: AsyncSession):
        """
        Initialize place service.

        Args:
            db: Database session
        """
        self.db = db
        self.gmaps = googlemaps.Client(key=settings.GOOGLE_PLACES_API_KEY)

    async def lookup_place(
        self,
        name: str,
        city: Optional[str] = None,
        address: Optional[str] = None,
        lat: Optional[float] = None,
        lng: Optional[float] = None
    ) -> Tuple[Place, bool]:
        """
        Look up a restaurant, creating it if not found.

        This method:
        1. Searches local database first
        2. If not found or cache expired, queries Google Places API
        3. Creates or updates place record
        4. Returns place and cache hit status

        Args:
            name: Restaurant name
            city: City location
            address: Full address
            lat: Latitude
            lng: Longitude

        Returns:
            Tuple of (Place object, cache_hit boolean)

        Example:
            >>> place_service = PlaceService(db)
            >>> place, cache_hit = await place_service.lookup_place("Joe's Pizza", "New York")
            >>> print(f"Found {place.name}, cache hit: {cache_hit}")
        """
        logger.info(f"Looking up place: {name} in {city}")

        # First, search local database
        existing_place = await self._find_existing_place(name, city, address)

        if existing_place and existing_place.is_cache_valid:
            logger.info(f"Found cached place: {existing_place.id}")
            return existing_place, True

        # Need to fetch from external APIs
        try:
            place_data = await self._search_google_places(name, city, address, lat, lng)

            if existing_place:
                # Update existing place
                updated_place = await self._update_place(existing_place, place_data)
                logger.info(f"Updated existing place: {updated_place.id}")
                return updated_place, False
            else:
                # Create new place
                new_place = await self._create_place(place_data)
                logger.info(f"Created new place: {new_place.id}")
                return new_place, False

        except Exception as e:
            logger.error(f"Failed to lookup place from external APIs: {str(e)}")

            # Return existing place even if cache expired, or raise if no existing
            if existing_place:
                logger.warning(f"Returning stale cached data for place: {existing_place.id}")
                return existing_place, True
            else:
                raise

    async def _find_existing_place(
        self,
        name: str,
        city: Optional[str] = None,
        address: Optional[str] = None
    ) -> Optional[Place]:
        """
        Search for existing place in database.

        Args:
            name: Restaurant name
            city: City location
            address: Full address

        Returns:
            Place object if found, None otherwise
        """
        # Build search query with fuzzy matching
        query = select(Place)

        # Primary search by name and city
        conditions = []

        # Name similarity (case-insensitive, partial match)
        name_condition = Place.name.ilike(f"%{name.strip()}%")
        conditions.append(name_condition)

        # City match if provided
        if city:
            city_condition = Place.city.ilike(f"%{city.strip()}%")
            conditions.append(city_condition)

        # Combine conditions
        if len(conditions) > 1:
            query = query.where(and_(*conditions))
        else:
            query = query.where(conditions[0])

        # Execute query
        result = await self.db.execute(query)
        places = result.scalars().all()

        if not places:
            return None

        # If multiple matches, pick the best one based on name similarity
        best_match = max(
            places,
            key=lambda p: self._calculate_name_similarity(name, p.name)
        )

        return best_match

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two restaurant names.

        Args:
            name1: First name
            name2: Second name

        Returns:
            Similarity score (0.0 to 1.0)
        """
        import difflib

        return difflib.SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

    async def _search_google_places(
        self,
        name: str,
        city: Optional[str] = None,
        address: Optional[str] = None,
        lat: Optional[float] = None,
        lng: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Search Google Places API for restaurant data.

        Args:
            name: Restaurant name
            city: City location
            address: Full address
            lat: Latitude
            lng: Longitude

        Returns:
            Place data from Google Places API

        Raises:
            Exception if no suitable place found
        """
        logger.info(f"Searching Google Places for: {name}")

        try:
            # Build search query
            if address:
                query = f"{name} {address}"
            elif city:
                query = f"{name} {city}"
            else:
                query = name

            # Add location bias if coordinates provided
            location_bias = None
            if lat and lng:
                location_bias = {"lat": lat, "lng": lng}

            # Search places
            places_result = self.gmaps.places(
                query=query,
                type="restaurant",
                location=location_bias,
                radius=5000 if location_bias else None
            )

            if not places_result.get("results"):
                raise Exception(f"No places found for query: {query}")

            # Pick the best match (first result is usually best)
            place = places_result["results"][0]

            # Get detailed place information
            place_id = place["place_id"]
            details = self.gmaps.place(
                place_id=place_id,
                fields=[
                    "name", "formatted_address", "geometry", "place_id",
                    "rating", "user_ratings_total", "price_level",
                    "formatted_phone_number", "website", "types"
                ]
            )

            place_details = details["result"]

            # Extract and normalize data
            geometry = place_details.get("geometry", {}).get("location", {})
            place_data = {
                "name": place_details.get("name", name),
                "address": place_details.get("formatted_address"),
                "lat": geometry.get("lat"),
                "lng": geometry.get("lng"),
                "google_place_id": place_details.get("place_id"),
                "phone": place_details.get("formatted_phone_number"),
                "website": place_details.get("website"),
                "rating": place_details.get("rating"),
                "review_count": place_details.get("user_ratings_total", 0),
                "price_level": place_details.get("price_level"),
                "types": place_details.get("types", []),
            }

            # Extract city from address
            if not city and place_data["address"]:
                place_data["city"] = self._extract_city_from_address(place_data["address"])
            else:
                place_data["city"] = city

            # Extract cuisine tags from types
            place_data["cuisine_tags"] = self._extract_cuisine_tags(place_data["types"])

            logger.info(f"Found Google place: {place_data['name']} (ID: {place_data['google_place_id']})")
            return place_data

        except Exception as e:
            logger.error(f"Google Places API error: {str(e)}")
            raise

    def _extract_city_from_address(self, address: str) -> Optional[str]:
        """
        Extract city name from formatted address.

        Args:
            address: Formatted address string

        Returns:
            City name if extractable
        """
        # Simple heuristic: city is usually the second-to-last component
        # Example: "123 Main St, New York, NY 10001, USA"
        parts = [part.strip() for part in address.split(",")]

        if len(parts) >= 2:
            # Return the component that looks like a city (not a state/ZIP)
            for part in parts[-3:-1]:  # Check a few parts before state
                if part and not part.isdigit() and len(part.split()) <= 3:
                    return part

        return None

    def _extract_cuisine_tags(self, place_types: List[str]) -> List[str]:
        """
        Extract cuisine-related tags from Google Places types.

        Args:
            place_types: List of place types from Google

        Returns:
            List of cuisine tags
        """
        cuisine_mapping = {
            "chinese_restaurant": "chinese",
            "japanese_restaurant": "japanese",
            "korean_restaurant": "korean",
            "thai_restaurant": "thai",
            "vietnamese_restaurant": "vietnamese",
            "indian_restaurant": "indian",
            "italian_restaurant": "italian",
            "mexican_restaurant": "mexican",
            "french_restaurant": "french",
            "american_restaurant": "american",
            "mediterranean_restaurant": "mediterranean",
            "greek_restaurant": "greek",
            "turkish_restaurant": "turkish",
            "middle_eastern_restaurant": "middle_eastern",
            "seafood_restaurant": "seafood",
            "steakhouse": "steakhouse",
            "barbecue_restaurant": "barbecue",
            "pizza_restaurant": "pizza",
            "bakery": "bakery",
            "cafe": "cafe",
            "fast_food_restaurant": "fast_food",
        }

        tags = []
        for place_type in place_types:
            if place_type in cuisine_mapping:
                tags.append(cuisine_mapping[place_type])

        # Add generic restaurant tag if no specific cuisine found
        if not tags and "restaurant" in place_types:
            tags.append("restaurant")

        return tags

    async def _create_place(self, place_data: Dict[str, Any]) -> Place:
        """
        Create a new place record.

        Args:
            place_data: Place data from external API

        Returns:
            Created Place object
        """
        # Set cache expiration
        cache_expires_at = datetime.utcnow() + timedelta(hours=settings.CACHE_TTL_HOURS)

        place = Place(
            name=place_data["name"],
            address=place_data.get("address"),
            city=place_data.get("city"),
            lat=place_data.get("lat"),
            lng=place_data.get("lng"),
            google_place_id=place_data.get("google_place_id"),
            phone=place_data.get("phone"),
            website=place_data.get("website"),
            rating=place_data.get("rating"),
            review_count=place_data.get("review_count", 0),
            price_level=place_data.get("price_level"),
            cuisine_tags=place_data.get("cuisine_tags", []),
            category="Restaurant",  # Default category
            cache_expires_at=cache_expires_at,
        )

        self.db.add(place)
        await self.db.commit()
        await self.db.refresh(place)

        logger.info(f"Created place: {place.name} (ID: {place.id})")
        return place

    async def _update_place(self, place: Place, place_data: Dict[str, Any]) -> Place:
        """
        Update an existing place with fresh data.

        Args:
            place: Existing place object
            place_data: Fresh data from external API

        Returns:
            Updated Place object
        """
        # Update fields
        place.name = place_data["name"]
        place.address = place_data.get("address") or place.address
        place.city = place_data.get("city") or place.city
        place.lat = place_data.get("lat") or place.lat
        place.lng = place_data.get("lng") or place.lng
        place.google_place_id = place_data.get("google_place_id") or place.google_place_id
        place.phone = place_data.get("phone") or place.phone
        place.website = place_data.get("website") or place.website
        place.rating = place_data.get("rating") or place.rating
        place.review_count = place_data.get("review_count", 0)
        place.price_level = place_data.get("price_level") or place.price_level
        place.cuisine_tags = place_data.get("cuisine_tags", []) or place.cuisine_tags

        # Update cache expiration
        place.cache_expires_at = datetime.utcnow() + timedelta(hours=settings.CACHE_TTL_HOURS)

        await self.db.commit()
        await self.db.refresh(place)

        logger.info(f"Updated place: {place.name} (ID: {place.id})")
        return place

    async def get_place_by_id(self, place_id: UUID) -> Optional[Place]:
        """
        Get a place by its ID.

        Args:
            place_id: UUID of the place

        Returns:
            Place object if found
        """
        query = select(Place).where(Place.id == place_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def search_places(
        self,
        query: str,
        city: Optional[str] = None,
        limit: int = 10
    ) -> List[Place]:
        """
        Search for places by name and location.

        Args:
            query: Search query for restaurant names
            city: Optional city filter
            limit: Maximum results to return

        Returns:
            List of matching places
        """
        # Build search query
        search_query = select(Place)

        conditions = []

        # Name search (case-insensitive)
        name_condition = Place.name.ilike(f"%{query}%")
        conditions.append(name_condition)

        # City filter if provided
        if city:
            city_condition = Place.city.ilike(f"%{city}%")
            conditions.append(city_condition)

        # Combine conditions
        if len(conditions) > 1:
            search_query = search_query.where(and_(*conditions))
        else:
            search_query = search_query.where(conditions[0])

        # Order by rating and limit
        search_query = search_query.order_by(Place.rating.desc()).limit(limit)

        # Execute query
        result = await self.db.execute(search_query)
        return result.scalars().all()

    async def update_cache_metadata(self, place: Place) -> None:
        """
        Update cache metadata for a place.

        Args:
            place: Place object to update
        """
        place.cache_expires_at = datetime.utcnow() + timedelta(hours=settings.CACHE_TTL_HOURS)
        place.reviews_last_fetched = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(place)