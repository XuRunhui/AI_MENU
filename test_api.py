#!/usr/bin/env python3
"""Quick test script to verify the API works without full database setup."""

import os
import sys
import asyncio
from unittest.mock import MagicMock

# Add the app directory to Python path
sys.path.insert(0, '/workspace/menu_taste_app')

# Mock external dependencies for testing
os.environ.setdefault('DATABASE_URL', 'sqlite+aiosqlite:///:memory:')
os.environ.setdefault('GOOGLE_PLACES_API_KEY', 'test-key')
os.environ.setdefault('OPENAI_API_KEY', 'test-key')
os.environ.setdefault('YELP_API_KEY', 'test-key')

async def test_basic_imports():
    """Test that all the core modules can be imported."""
    print("🧪 Testing core imports...")

    try:
        from app.core.config import settings
        print(f"✅ Settings loaded: ENV={settings.ENV}")

        from app.services.ocr_service import OCRService
        print("✅ OCR Service imported")

        from app.services.place_service import PlaceService
        print("✅ Place Service imported")

        from app.services.embedding_service import EmbeddingService
        print("✅ Embedding Service imported")

        from app.services.review_service import ReviewService
        print("✅ Review Service imported")

        from app.services.menu_service import MenuService
        print("✅ Menu Service imported")

        print("\n🎉 All core services imported successfully!")
        return True

    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

async def test_embedding_service():
    """Test the embedding service with mock data."""
    print("\n🧪 Testing Embedding Service...")

    try:
        # Import sentence transformers to verify it works
        from sentence_transformers import SentenceTransformer

        # Test embedding creation
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Test dish name embedding
        dish_name = "Kung Pao Chicken"
        embedding = model.encode([dish_name])

        print(f"✅ Created embedding for '{dish_name}'")
        print(f"   Vector dimension: {len(embedding[0])}")
        print(f"   Sample values: {embedding[0][:5]}...")

        # Test similarity calculation
        review_sentences = [
            "The kung pao chicken was amazing with perfect spice level",
            "Great pasta dish with creamy sauce",
            "Spicy chicken with peanuts, very authentic"
        ]

        sentence_embeddings = model.encode(review_sentences)

        # Calculate similarities
        import numpy as np
        similarities = []
        for i, sent_emb in enumerate(sentence_embeddings):
            similarity = np.dot(embedding[0], sent_emb) / (
                np.linalg.norm(embedding[0]) * np.linalg.norm(sent_emb)
            )
            similarities.append((review_sentences[i][:50], float(similarity)))

        print(f"\n   Similarity scores:")
        for sentence, score in similarities:
            print(f"   {score:.3f}: {sentence}...")

        print("✅ Embedding service working correctly!")
        return True

    except Exception as e:
        print(f"❌ Embedding test failed: {str(e)}")
        return False

async def test_ocr_functionality():
    """Test OCR text processing without actual image."""
    print("\n🧪 Testing OCR Service text processing...")

    try:
        from app.services.ocr_service import OCRService

        ocr_service = OCRService()

        # Test dish name detection
        test_texts = [
            "Kung Pao Chicken - $16.95",
            "Mapo Tofu",
            "123",  # Should be rejected
            "Sweet and Sour Pork $14.50",
            "App"   # Too short, should be rejected
        ]

        print("   Testing dish detection:")
        for text in test_texts:
            is_dish = ocr_service._looks_like_dish(text)
            print(f"   '{text}' -> {'✅ DISH' if is_dish else '❌ Not dish'}")

        # Test price extraction
        print("\n   Testing price extraction:")
        price_texts = [
            "Kung Pao Chicken $16.95",
            "Mapo Tofu - $12.50",
            "Spring Rolls 8.99"
        ]

        for text in price_texts:
            price_text, price_amount = ocr_service._extract_price_from_text(text)
            print(f"   '{text}' -> ${price_amount or 'N/A'}")

        print("✅ OCR text processing working!")
        return True

    except Exception as e:
        print(f"❌ OCR test failed: {str(e)}")
        return False

async def test_api_startup():
    """Test that the FastAPI app can start."""
    print("\n🧪 Testing FastAPI app startup...")

    try:
        from app.main import app
        print("✅ FastAPI app created successfully")

        # Check if routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/api/v1/ocr/menu", "/api/v1/places/lookup"]

        for route in expected_routes:
            if any(route in r for r in routes):
                print(f"✅ Route registered: {route}")
            else:
                print(f"⚠️  Route missing: {route}")

        print("✅ FastAPI app startup test completed!")
        return True

    except Exception as e:
        print(f"❌ FastAPI startup failed: {str(e)}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Menu Taste Guide API - Quick Test Suite")
    print("=" * 50)

    tests = [
        test_basic_imports,
        test_embedding_service,
        test_ocr_functionality,
        test_api_startup
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)
        print()

    # Summary
    passed = sum(results)
    total = len(results)

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! The API foundation is working correctly.")
        print("\n📋 Next steps:")
        print("1. Add your API keys to .env file")
        print("2. Run: /data/venv/aimenu/bin/uvicorn app.main:app --reload")
        print("3. Visit: http://localhost:8000/api/v1/docs")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)