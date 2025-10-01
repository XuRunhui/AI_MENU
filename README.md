# Menu Taste Guide API

A taste-centric menu recommendation system that helps diners understand what restaurant dishes actually taste like and suggests balanced meal combinations.

## 🍽️ What It Does

**Core Features:**
1. **OCR Menu Processing**: Extract dish names and prices from menu photos using Google Vision API
2. **Restaurant Lookup**: Find restaurants and fetch reviews from Google Places/Yelp APIs
3. **Semantic Matching**: Match dishes to review mentions using embeddings (sentence-transformers)
4. **Taste Analysis**: Extract taste aspects (spice, texture, richness) using rule-based ABSA
5. **LLM Generation**: Create human-readable taste descriptions with confidence levels
6. **Combo Recommendations**: Suggest balanced dish combinations based on party size and preferences

## 🏗️ Project Structure

```
menu_taste_app/
├── app/
│   ├── api/v1/endpoints/          # API endpoints
│   │   ├── ocr.py                 # Menu OCR processing
│   │   ├── places.py              # Restaurant lookup
│   │   ├── taste_cards.py         # Taste descriptions
│   │   └── combos.py              # Dish combinations
│   ├── core/                      # Core configuration
│   │   ├── config.py              # Settings and environment
│   │   └── logging.py             # Logging setup
│   ├── db/                        # Database setup
│   │   └── base.py                # SQLAlchemy async setup
│   ├── models/                    # Database models
│   │   ├── place.py               # Restaurant model
│   │   ├── menu_item.py           # Menu dish model
│   │   ├── review.py              # Review and embeddings
│   │   ├── taste_card.py          # Generated descriptions
│   │   └── combo.py               # Combo recommendations
│   ├── schemas/                   # Pydantic request/response schemas
│   ├── services/                  # Business logic layer
│   │   ├── ocr_service.py         # Google Vision OCR
│   │   ├── place_service.py       # Restaurant lookup
│   │   ├── review_service.py      # Review fetching
│   │   ├── embedding_service.py   # Semantic matching
│   │   ├── menu_service.py        # Menu processing
│   │   └── [more services...]
│   └── main.py                    # FastAPI application
├── alembic/                       # Database migrations
├── requirements.txt               # Python dependencies
└── .env.example                   # Environment variables template
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and enter directory
cd menu_taste_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys:
# - GOOGLE_PLACES_API_KEY=your-google-api-key
# - GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
# - YELP_API_KEY=your-yelp-api-key
# - OPENAI_API_KEY=your-openai-api-key
# - DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/menu_db
```

### 3. Database Setup

```bash
# Start PostgreSQL (using Docker)
docker run --name menu-postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=menu_taste_db -p 5432:5432 -d postgres:15

# Run migrations
alembic upgrade head
```

### 4. Start the API

```bash
# Development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/api/v1/docs
```

## 📋 API Usage Examples

### 1. Process Menu Images
```python
import requests

# Upload menu photos
files = [("images", open("menu_page1.jpg", "rb"))]
data = {"restaurant_name": "Szechuan Chef", "city": "Seattle"}

response = requests.post(
    "http://localhost:8000/api/v1/ocr/menu",
    files=files,
    data=data
)

menu_items = response.json()["menu_items"]
print(f"Extracted {len(menu_items)} dishes")
```

### 2. Look Up Restaurant
```python
# Find restaurant and fetch reviews
place_data = {
    "restaurant_name": "Szechuan Chef",
    "city": "Seattle"
}

response = requests.post(
    "http://localhost:8000/api/v1/places/lookup",
    json=place_data
)

place = response.json()["place"]
place_id = place["id"]
print(f"Found {place['name']}, fetched {response.json()['reviews_fetched']} reviews")
```

### 3. Generate Taste Card
```python
# Generate taste description for a dish
taste_request = {
    "place_id": place_id,
    "dish_name": "Mapo Tofu"
}

response = requests.post(
    "http://localhost:8000/api/v1/taste-cards/generate",
    json=taste_request
)

taste_card = response.json()
print("Taste Description:")
for bullet in taste_card["bullets"]:
    print(f"• {bullet}")
```

### 4. Get Combo Recommendations
```python
# Get balanced meal combinations
combo_request = {
    "place_id": place_id,
    "party_size": 3,
    "budget": 75.0,
    "dietary_constraints": ["vegetarian"],
    "spice_preference": 2
}

response = requests.post(
    "http://localhost:8000/api/v1/combos/recommend",
    json=combo_request
)

combos = response.json()["combos"]
for combo in combos:
    print(f"Combo (Score: {combo['score']}):")
    for item in combo["items"]:
        print(f"  - {item['name']} ({item['role']})")
    print(f"Rationale: {combo['rationale']}")
```

## 🔧 Key Algorithms Implemented

### 1. Dish-to-Review Matching
```python
# Core semantic similarity matching
dish_embedding = model.encode(["Kung Pao Chicken"])
review_sentences = ["The kung pao had perfect numbing spice...", ...]

similarities = []
for sentence in review_sentences:
    sentence_embedding = model.encode([sentence])
    similarity = cosine_similarity(dish_embedding, sentence_embedding)
    if similarity > 0.56:  # Threshold
        similarities.append((sentence, similarity))

# Return top matches
top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:12]
```

### 2. Taste Aspect Extraction (ABSA)
```python
# Rule-based aspect extraction
aspects = {"spice": 0, "texture": [], "allergens": set()}

for sentence in review_snippets:
    if re.search(r"(numb|mala|peppercorn)", sentence, re.I):
        aspects["spice"] = 3
        aspects["heat_type"] = "peppercorn"

    if "crispy" in sentence.lower():
        aspects["texture"].append("crispy")
```

### 3. Combo Scoring Algorithm
```python
# Balance-based combo scoring
def score_combo(dishes, constraints):
    score = sum(dish.sentiment_score for dish in dishes)

    # Balance bonuses
    spice_levels = [d.aspects["spice"] for d in dishes]
    if len(set(spice_levels)) > 1:  # Variety in spice
        score += 10

    # Constraint penalties
    if any(allergen in d.aspects["allergens"] for d in dishes
           for allergen in constraints["allergies"]):
        score -= 100

    return score
```

## 🔄 Development Workflow

### Phase 1: MVP (Weeks 1-2) ✅
- [x] FastAPI project structure
- [x] Database models (PostgreSQL + Alembic)
- [x] Pydantic schemas for all endpoints
- [x] OCR service (Google Vision + Tesseract fallback)
- [x] Place lookup (Google Places API)
- [x] Review fetching (Google + Yelp)
- [x] Embedding service (sentence-transformers)

### Phase 2: Core Features (Weeks 3-4)
- [ ] Complete ABSA service implementation
- [ ] LLM taste card generation service
- [ ] Combo scoring algorithm
- [ ] Redis caching layer
- [ ] Comprehensive test suite

### Phase 3: Enhancement (Weeks 5-6)
- [ ] Fallback hierarchy (city-wide → global knowledge)
- [ ] Confidence scoring improvements
- [ ] Batch processing endpoints
- [ ] Performance optimization

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app tests/

# Test specific service
pytest tests/test_embedding_service.py -v
```

## 📊 Database Schema

### Key Tables
- **places**: Restaurant data (Google Place ID, location, cuisine tags)
- **menu_items**: Extracted dishes (OCR results, normalized names)
- **reviews**: Fetched review text (Google, Yelp sources)
- **review_embeds**: Sentence embeddings for semantic search
- **taste_cards**: Generated taste descriptions (aspects, bullets, confidence)
- **combo_templates**: Saved dish combinations (scoring, constraints)

## 🚀 Production Deployment

```bash
# Build Docker image
docker build -t menu-taste-api .

# Run with docker-compose
docker-compose up -d

# Includes: FastAPI app, PostgreSQL, Redis
```

## 🎯 Next Steps

1. **Complete the remaining services**: ABSA, Taste Card Generation, Combo Scoring
2. **Add Redis caching** for performance optimization
3. **Implement comprehensive tests** for all services
4. **Add authentication** and rate limiting
5. **Build the frontend** (React Native app for camera integration)

## 🤝 Contributing

The codebase follows these patterns:
- **Service Layer**: Business logic isolated in `/services/`
- **Type Safety**: Full type hints and Pydantic validation
- **Async/Await**: All I/O operations are asynchronous
- **Error Handling**: Comprehensive logging and error responses
- **Documentation**: Docstrings with examples for all functions

Ready to continue building! The foundation is solid and ready for the remaining features. 🎉