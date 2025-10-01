"""Review service for fetching and processing restaurant reviews."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from uuid import UUID

import aiohttp
import googlemaps
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.config import settings
from app.models.place import Place
from app.models.review import Review
from app.services.embedding_service import EmbeddingService


class ReviewService:
    """Service for fetching and managing restaurant reviews."""

    def __init__(self, db: AsyncSession):
        """
        Initialize review service.

        Args:
            db: Database session
        """
        self.db = db
        self.gmaps = googlemaps.Client(key=settings.GOOGLE_PLACES_API_KEY)

    async def fetch_and_store_reviews(
        self,
        place: Place,
        force_refresh: bool = False
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Fetch reviews from external APIs and store them in database.

        This method:
        1. Checks if we need to fetch fresh reviews
        2. Fetches from Google Places and Yelp (if available)
        3. Stores new reviews in database
        4. Triggers embedding processing

        Args:
            place: Place object to fetch reviews for
            force_refresh: Whether to fetch even if cache is valid

        Returns:
            Tuple of (reviews_count, processing_stats)

        Example:
            >>> review_service = ReviewService(db)
            >>> count, stats = await review_service.fetch_and_store_reviews(place)
            >>> print(f"Fetched {count} reviews in {stats['processing_time_ms']}ms")
        """
        logger.info(f"Fetching reviews for place: {place.name} (ID: {place.id})")

        # Check if we need to fetch
        if not force_refresh and place.is_cache_valid:
            logger.info("Reviews are still cached and valid, skipping fetch")
            return 0, {"cache_hit": True}

        processing_stats = {
            "api_calls_made": 0,
            "google_reviews": 0,
            "yelp_reviews": 0,
            "processing_time_ms": 0,
            "review_languages": set(),
            "average_rating": 0.0,
            "cache_hit": False
        }

        start_time = datetime.utcnow()

        try:
            # Fetch from multiple sources concurrently
            tasks = []

            # Google Places reviews
            if place.google_place_id:
                tasks.append(self._fetch_google_reviews(place.google_place_id))

            # Yelp reviews (if available)
            if place.yelp_business_id:
                tasks.append(self._fetch_yelp_reviews(place.yelp_business_id))

            # Execute fetches concurrently
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                processing_stats["api_calls_made"] = len([r for r in results if not isinstance(r, Exception)])
            else:
                results = []

            # Process results
            all_reviews = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Review fetch task {i} failed: {str(result)}")
                    continue

                source_name = "google" if i == 0 else "yelp"
                reviews = result.get("reviews", [])
                all_reviews.extend(reviews)

                if source_name == "google":
                    processing_stats["google_reviews"] = len(reviews)
                else:
                    processing_stats["yelp_reviews"] = len(reviews)

            # Store new reviews in database
            stored_count = await self._store_reviews(place.id, all_reviews)

            # Calculate processing stats
            if all_reviews:
                processing_stats["review_languages"] = list(set(
                    r.get("language", "en") for r in all_reviews
                ))
                ratings = [r.get("rating") for r in all_reviews if r.get("rating")]
                if ratings:
                    processing_stats["average_rating"] = sum(ratings) / len(ratings)

            # Update place cache metadata
            place.reviews_last_fetched = datetime.utcnow()
            place.cache_expires_at = datetime.utcnow() + timedelta(hours=settings.CACHE_TTL_HOURS)
            await self.db.commit()

            # Process embeddings asynchronously
            if stored_count > 0:
                try:
                    embedding_service = EmbeddingService(self.db)
                    await embedding_service.process_reviews_for_embeddings(place.id)
                except Exception as e:
                    logger.error(f"Failed to process embeddings: {str(e)}")

            # Calculate processing time
            end_time = datetime.utcnow()
            processing_stats["processing_time_ms"] = int(
                (end_time - start_time).total_seconds() * 1000
            )

            logger.info(f"Fetched and stored {stored_count} reviews for place {place.id}")
            return stored_count, processing_stats

        except Exception as e:
            logger.error(f"Failed to fetch reviews for place {place.id}: {str(e)}")
            raise

    async def _fetch_google_reviews(self, place_id: str) -> Dict[str, Any]:
        """
        Fetch reviews from Google Places API.

        Args:
            place_id: Google Place ID

        Returns:
            Dictionary with reviews and metadata
        """
        logger.info(f"Fetching Google reviews for place_id: {place_id}")

        try:
            # Get place details with reviews
            details = self.gmaps.place(
                place_id=place_id,
                fields=["reviews", "rating", "user_ratings_total"]
            )

            place_details = details.get("result", {})
            raw_reviews = place_details.get("reviews", [])

            # Convert to our format
            reviews = []
            for review in raw_reviews:
                reviews.append({
                    "source": "google",
                    "external_id": review.get("time", str(review.get("author_name", ""))),
                    "author_name": review.get("author_name"),
                    "rating": review.get("rating"),
                    "text": review.get("text", ""),
                    "language": review.get("language", "en"),
                    "review_date": datetime.fromtimestamp(review.get("time", 0)) if review.get("time") else None
                })

            logger.info(f"Fetched {len(reviews)} Google reviews")
            return {
                "reviews": reviews,
                "total_rating": place_details.get("rating"),
                "total_review_count": place_details.get("user_ratings_total", 0)
            }

        except Exception as e:
            logger.error(f"Failed to fetch Google reviews: {str(e)}")
            return {"reviews": []}

    async def _fetch_yelp_reviews(self, business_id: str) -> Dict[str, Any]:
        """
        Fetch reviews from Yelp Fusion API.

        Args:
            business_id: Yelp business ID

        Returns:
            Dictionary with reviews and metadata
        """
        logger.info(f"Fetching Yelp reviews for business_id: {business_id}")

        if not settings.YELP_API_KEY:
            logger.warning("Yelp API key not configured, skipping Yelp reviews")
            return {"reviews": []}

        try:
            # Yelp API endpoint
            url = f"https://api.yelp.com/v3/businesses/{business_id}/reviews"
            headers = {
                "Authorization": f"Bearer {settings.YELP_API_KEY}",
                "User-Agent": "MenuTasteApp/1.0"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        raw_reviews = data.get("reviews", [])

                        # Convert to our format
                        reviews = []
                        for review in raw_reviews:
                            review_date = None
                            if review.get("time_created"):
                                try:
                                    review_date = datetime.fromisoformat(
                                        review["time_created"].replace("Z", "+00:00")
                                    )
                                except:
                                    pass

                            reviews.append({
                                "source": "yelp",
                                "external_id": review.get("id"),
                                "author_name": review.get("user", {}).get("name"),
                                "rating": review.get("rating"),
                                "text": review.get("text", ""),
                                "language": "en",  # Yelp doesn't specify language
                                "review_date": review_date
                            })

                        logger.info(f"Fetched {len(reviews)} Yelp reviews")
                        return {"reviews": reviews}

                    else:
                        logger.warning(f"Yelp API returned status {response.status}")
                        return {"reviews": []}

        except Exception as e:
            logger.error(f"Failed to fetch Yelp reviews: {str(e)}")
            return {"reviews": []}

    async def _store_reviews(
        self,
        place_id: UUID,
        reviews_data: List[Dict[str, Any]]
    ) -> int:
        """
        Store reviews in the database, avoiding duplicates.

        Args:
            place_id: UUID of the place
            reviews_data: List of review dictionaries

        Returns:
            Number of new reviews stored
        """
        if not reviews_data:
            return 0

        stored_count = 0

        for review_data in reviews_data:
            try:
                # Check if review already exists
                existing = await self._find_existing_review(
                    place_id,
                    review_data["source"],
                    review_data.get("external_id")
                )

                if existing:
                    logger.debug(f"Review {review_data.get('external_id')} already exists, skipping")
                    continue

                # Create new review
                review = Review(
                    place_id=place_id,
                    source=review_data["source"],
                    external_id=review_data.get("external_id"),
                    author_name=review_data.get("author_name"),
                    rating=review_data.get("rating"),
                    text=review_data.get("text", ""),
                    language=review_data.get("language", "en"),
                    review_date=review_data.get("review_date"),
                    is_processed="pending"
                )

                self.db.add(review)
                stored_count += 1

            except Exception as e:
                logger.error(f"Failed to store review: {str(e)}")
                continue

        # Commit all reviews
        if stored_count > 0:
            await self.db.commit()

        logger.info(f"Stored {stored_count} new reviews")
        return stored_count

    async def _find_existing_review(
        self,
        place_id: UUID,
        source: str,
        external_id: Optional[str]
    ) -> Optional[Review]:
        """
        Find existing review by place, source, and external ID.

        Args:
            place_id: UUID of the place
            source: Review source ("google", "yelp")
            external_id: External review ID

        Returns:
            Existing Review if found
        """
        if not external_id:
            return None

        query = select(Review).where(
            and_(
                Review.place_id == place_id,
                Review.source == source,
                Review.external_id == external_id
            )
        )

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_place_reviews(
        self,
        place_id: UUID,
        limit: int = 50,
        source: Optional[str] = None
    ) -> List[Review]:
        """
        Get reviews for a place.

        Args:
            place_id: UUID of the place
            limit: Maximum number of reviews
            source: Optional source filter

        Returns:
            List of reviews
        """
        query = select(Review).where(Review.place_id == place_id)

        if source:
            query = query.where(Review.source == source)

        query = query.order_by(Review.review_date.desc()).limit(limit)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_review_stats(self, place_id: UUID) -> Dict[str, Any]:
        """
        Get review statistics for a place.

        Args:
            place_id: UUID of the place

        Returns:
            Dictionary with review statistics
        """
        # Count reviews by source
        source_counts = await self.db.execute(
            select(Review.source, func.count(Review.id))
            .where(Review.place_id == place_id)
            .group_by(Review.source)
        )

        source_stats = dict(source_counts.fetchall())

        # Get average rating
        avg_rating = await self.db.execute(
            select(func.avg(Review.rating))
            .where(Review.place_id == place_id)
        )

        average_rating = avg_rating.scalar()

        # Get processing status counts
        processing_counts = await self.db.execute(
            select(Review.is_processed, func.count(Review.id))
            .where(Review.place_id == place_id)
            .group_by(Review.is_processed)
        )

        processing_stats = dict(processing_counts.fetchall())

        return {
            "total_reviews": sum(source_stats.values()),
            "by_source": source_stats,
            "average_rating": float(average_rating) if average_rating else None,
            "processing_status": processing_stats
        }