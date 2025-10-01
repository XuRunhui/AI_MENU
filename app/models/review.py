"""Review and review embedding models."""

from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class Review(Base):
    """Customer review model."""

    __tablename__ = "reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    place_id = Column(UUID(as_uuid=True), ForeignKey("places.id"), nullable=False, index=True)

    # Review metadata
    source = Column(String(50), nullable=False)  # "google", "yelp"
    external_id = Column(String(255), index=True)  # Original review ID from source
    author_name = Column(String(255))
    rating = Column(Float)  # 1-5 star rating

    # Review content
    text = Column(Text, nullable=False)
    language = Column(String(10), default="en")

    # Timestamps
    review_date = Column(DateTime(timezone=True))  # When review was written
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Processing metadata
    is_processed = Column(String(20), default="pending")  # "pending", "processed", "failed"
    sentence_count = Column(Integer, default=0)

    # Relationships
    place = relationship("Place", back_populates="reviews")
    embeddings = relationship("ReviewEmbed", back_populates="review", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_review_source_external_id", "source", "external_id"),
        Index("idx_review_place_rating", "place_id", "rating"),
    )

    def __repr__(self) -> str:
        return f"<Review(source='{self.source}', rating={self.rating}, place_id='{self.place_id}')>"


class ReviewEmbed(Base):
    """Review sentence embeddings for semantic search."""

    __tablename__ = "review_embeds"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    review_id = Column(UUID(as_uuid=True), ForeignKey("reviews.id"), nullable=False, index=True)

    # Sentence info
    sentence_index = Column(Integer, nullable=False)  # Position in review (0, 1, 2...)
    text_span = Column(Text, nullable=False)  # The actual sentence text
    char_start = Column(Integer)  # Start position in original review
    char_end = Column(Integer)  # End position in original review

    # Embedding vector (stored as array of floats)
    vector = Column(ARRAY(Float), nullable=False)  # 384-dim for all-MiniLM-L6-v2

    # Extracted aspects (cached from ABSA)
    aspects_json = Column(JSON, default={})  # Cached aspect scores for this sentence

    # Metadata
    model_name = Column(String(100), default="all-MiniLM-L6-v2")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    review = relationship("Review", back_populates="embeddings")

    # Indexes for similarity search
    __table_args__ = (
        Index("idx_review_embed_review_sentence", "review_id", "sentence_index"),
    )

    def __repr__(self) -> str:
        return f"<ReviewEmbed(review_id='{self.review_id}', sentence_index={self.sentence_index})>"

    @property
    def cosine_similarity(self, other_vector: list[float]) -> float:
        """Calculate cosine similarity with another vector."""
        import numpy as np

        a = np.array(self.vector)
        b = np.array(other_vector)

        # Avoid division by zero
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)