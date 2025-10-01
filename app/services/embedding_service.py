"""Embedding service for semantic similarity matching."""

import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.config import settings
from app.models.review import Review, ReviewEmbed
from app.models.place import Place


class EmbeddingService:
    """Service for creating and matching sentence embeddings."""

    def __init__(self, db: AsyncSession):
        """
        Initialize embedding service.

        Args:
            db: Database session
        """
        self.db = db
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.vector_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Initialized embedding service with model: {settings.EMBEDDING_MODEL}")
        logger.info(f"Vector dimension: {self.vector_dim}")

    async def process_reviews_for_embeddings(
        self,
        place_id: UUID,
        force_reprocess: bool = False
    ) -> int:
        """
        Process all reviews for a place to create sentence embeddings.

        This method:
        1. Fetches unprocessed reviews for the place
        2. Splits reviews into sentences
        3. Creates embeddings for each sentence
        4. Stores embeddings in the database

        Args:
            place_id: UUID of the place
            force_reprocess: Whether to reprocess already processed reviews

        Returns:
            Number of embeddings created

        Example:
            >>> embedding_service = EmbeddingService(db)
            >>> count = await embedding_service.process_reviews_for_embeddings(place_id)
            >>> print(f"Created {count} embeddings")
        """
        logger.info(f"Processing reviews for embeddings: place_id={place_id}")

        # Fetch reviews that need processing
        query = select(Review).where(Review.place_id == place_id)

        if not force_reprocess:
            query = query.where(Review.is_processed != "processed")

        result = await self.db.execute(query)
        reviews = result.scalars().all()

        if not reviews:
            logger.info("No reviews found to process")
            return 0

        total_embeddings = 0

        for review in reviews:
            try:
                logger.debug(f"Processing review {review.id}")

                # Split review into sentences
                sentences = self._split_into_sentences(review.text)

                if not sentences:
                    continue

                # Create embeddings for all sentences
                embeddings = self.model.encode(sentences)

                # Store embeddings in database
                embeddings_created = await self._store_review_embeddings(
                    review,
                    sentences,
                    embeddings
                )

                total_embeddings += embeddings_created

                # Mark review as processed
                review.is_processed = "processed"
                review.sentence_count = len(sentences)

            except Exception as e:
                logger.error(f"Failed to process review {review.id}: {str(e)}")
                review.is_processed = "failed"

        # Commit all changes
        await self.db.commit()

        logger.info(f"Created {total_embeddings} embeddings for {len(reviews)} reviews")
        return total_embeddings

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split review text into sentences.

        Args:
            text: Review text

        Returns:
            List of sentence strings

        Example:
            >>> sentences = self._split_into_sentences("Great food! Service was slow.")
            >>> print(sentences)  # ["Great food!", "Service was slow."]
        """
        if not text:
            return []

        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())

        # Simple sentence splitting using regex
        # This could be improved with spaCy or NLTK for better accuracy
        sentences = re.split(r'[.!?]+', text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()

            # Skip very short sentences or just punctuation
            if len(sentence) < 10 or not re.search(r'[a-zA-Z]', sentence):
                continue

            # Skip sentences that are mostly numbers/symbols
            word_count = len(re.findall(r'\b\w+\b', sentence))
            if word_count < 3:
                continue

            cleaned_sentences.append(sentence)

        return cleaned_sentences

    async def _store_review_embeddings(
        self,
        review: Review,
        sentences: List[str],
        embeddings: np.ndarray
    ) -> int:
        """
        Store sentence embeddings in the database.

        Args:
            review: Review object
            sentences: List of sentences
            embeddings: Numpy array of embeddings

        Returns:
            Number of embeddings stored
        """
        embeddings_created = 0

        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            try:
                # Find character positions in original text
                char_start, char_end = self._find_sentence_position(review.text, sentence)

                # Create embedding record
                review_embed = ReviewEmbed(
                    review_id=review.id,
                    sentence_index=i,
                    text_span=sentence,
                    char_start=char_start,
                    char_end=char_end,
                    vector=embedding.tolist(),  # Convert numpy array to list
                    model_name=settings.EMBEDDING_MODEL
                )

                self.db.add(review_embed)
                embeddings_created += 1

            except Exception as e:
                logger.error(f"Failed to store embedding for sentence {i}: {str(e)}")
                continue

        return embeddings_created

    def _find_sentence_position(self, full_text: str, sentence: str) -> Tuple[int, int]:
        """
        Find the character position of a sentence in the full text.

        Args:
            full_text: Full review text
            sentence: Sentence to find

        Returns:
            Tuple of (start_position, end_position)
        """
        # Simple substring search (could be improved)
        start = full_text.find(sentence)
        if start >= 0:
            return start, start + len(sentence)
        else:
            # Fallback: return approximate position
            return 0, len(sentence)

    async def find_matching_sentences(
        self,
        dish_name: str,
        place_id: UUID,
        max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Find review sentences that mention or relate to a specific dish.

        This is the core matching algorithm:
        1. Create embedding for the dish name
        2. Calculate cosine similarity with all review sentence embeddings
        3. Return top matches above similarity threshold

        Args:
            dish_name: Name of the dish to search for
            place_id: UUID of the restaurant
            max_results: Maximum number of results (defaults to MAX_REVIEW_SNIPPETS)

        Returns:
            List of matching sentences with similarity scores and metadata

        Example:
            >>> matches = await embedding_service.find_matching_sentences("Kung Pao Chicken", place_id)
            >>> for match in matches:
            ...     print(f"Score: {match['similarity']:.3f} - {match['sentence']}")
        """
        if max_results is None:
            max_results = settings.MAX_REVIEW_SNIPPETS

        logger.info(f"Finding sentences matching '{dish_name}' at place {place_id}")

        # Create embedding for the dish name
        dish_embedding = self.model.encode([dish_name])[0]

        # Fetch all review embeddings for this place
        query = select(ReviewEmbed).join(Review).where(
            and_(
                Review.place_id == place_id,
                Review.is_processed == "processed"
            )
        )

        result = await self.db.execute(query)
        review_embeds = result.scalars().all()

        if not review_embeds:
            logger.warning(f"No review embeddings found for place {place_id}")
            return []

        # Calculate similarities
        matches = []
        for embed in review_embeds:
            try:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(dish_embedding, embed.vector)

                # Only include matches above threshold
                if similarity >= settings.SIMILARITY_THRESHOLD:
                    matches.append({
                        "sentence": embed.text_span,
                        "similarity": float(similarity),
                        "review_id": str(embed.review_id),
                        "sentence_index": embed.sentence_index,
                        "review_embed_id": str(embed.id),
                        "char_start": embed.char_start,
                        "char_end": embed.char_end
                    })

            except Exception as e:
                logger.error(f"Error calculating similarity for embed {embed.id}: {str(e)}")
                continue

        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        # Limit results
        matches = matches[:max_results]

        logger.info(f"Found {len(matches)} matching sentences for '{dish_name}'")
        return matches

    def _cosine_similarity(self, vec1: np.ndarray, vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector (numpy array)
            vec2: Second vector (list of floats)

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            # Avoid division by zero
            if norm_a == 0 or norm_b == 0:
                return 0.0

            similarity = dot_product / (norm_a * norm_b)

            # Ensure result is in valid range
            return max(0.0, min(1.0, float(similarity)))

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0

    async def get_similar_dishes(
        self,
        dish_name: str,
        place_id: Optional[UUID] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find dishes similar to the given dish name.

        This can be used for:
        1. Finding alternative dish names (aliases)
        2. Suggesting similar dishes to users
        3. Expanding search when no direct matches found

        Args:
            dish_name: Name of the dish
            place_id: Optional place filter
            limit: Maximum number of similar dishes

        Returns:
            List of similar dishes with similarity scores
        """
        logger.info(f"Finding dishes similar to '{dish_name}'")

        # Create embedding for input dish
        dish_embedding = self.model.encode([dish_name])[0]

        # This would require a dish embeddings table or search across all review embeddings
        # For now, return empty list (could be implemented later)
        logger.warning("Similar dish search not yet implemented")
        return []

    async def batch_process_place_reviews(
        self,
        place_ids: List[UUID],
        max_concurrent: int = 3
    ) -> Dict[UUID, int]:
        """
        Process reviews for multiple places concurrently.

        Args:
            place_ids: List of place UUIDs
            max_concurrent: Maximum concurrent processing tasks

        Returns:
            Dictionary mapping place_id to number of embeddings created
        """
        logger.info(f"Batch processing reviews for {len(place_ids)} places")

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_place(place_id: UUID) -> Tuple[UUID, int]:
            async with semaphore:
                try:
                    count = await self.process_reviews_for_embeddings(place_id)
                    return place_id, count
                except Exception as e:
                    logger.error(f"Failed to process place {place_id}: {str(e)}")
                    return place_id, 0

        # Process all places concurrently
        tasks = [process_single_place(place_id) for place_id in place_ids]
        results = await asyncio.gather(*tasks)

        # Convert to dictionary
        results_dict = dict(results)

        total_embeddings = sum(results_dict.values())
        logger.info(f"Batch processing complete: {total_embeddings} total embeddings created")

        return results_dict

    async def cleanup_old_embeddings(
        self,
        place_id: Optional[UUID] = None,
        days_old: int = 30
    ) -> int:
        """
        Clean up old embeddings to free database space.

        Args:
            place_id: Optional place filter
            days_old: Delete embeddings older than this many days

        Returns:
            Number of embeddings deleted
        """
        # This would implement cleanup logic based on created_at timestamps
        # For now, just log the intent
        logger.info(f"Cleanup old embeddings for place {place_id}, older than {days_old} days")
        return 0