#!/usr/bin/env python3
"""
Demo script showing the Menu Taste Guide API functionality.
This demonstrates the core algorithms without external dependencies.
"""

import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MenuItemDemo:
    """Demo menu item for testing."""
    name: str
    price: float
    section: str


@dataclass
class ReviewDemo:
    """Demo review for testing."""
    text: str
    rating: float
    author: str


class MenuTasteGuideDemo:
    """Demonstration of the core Menu Taste Guide algorithms."""

    def __init__(self):
        """Initialize with demo data."""
        self.demo_restaurant = "Szechuan Chef"
        self.demo_menu = self._create_demo_menu()
        self.demo_reviews = self._create_demo_reviews()

    def _create_demo_menu(self) -> List[MenuItemDemo]:
        """Create sample menu items."""
        return [
            MenuItemDemo("Mapo Tofu", 14.95, "Main Dishes"),
            MenuItemDemo("Kung Pao Chicken", 16.95, "Main Dishes"),
            MenuItemDemo("Dry-Fried Green Beans", 12.95, "Vegetables"),
            MenuItemDemo("Dan Dan Noodles", 13.95, "Noodles"),
            MenuItemDemo("Hot and Sour Soup", 8.95, "Soups"),
            MenuItemDemo("Steamed Rice", 3.95, "Sides"),
            MenuItemDemo("Sesame Ice Cream", 6.95, "Desserts"),
        ]

    def _create_demo_reviews(self) -> List[ReviewDemo]:
        """Create sample reviews."""
        return [
            ReviewDemo(
                "The mapo tofu was incredible! Very spicy with numbing Sichuan peppercorns. "
                "Silky tofu in rich red oil sauce. Authentic and mouth-numbingly delicious.",
                5.0, "Sarah M."
            ),
            ReviewDemo(
                "Kung pao chicken was perfectly balanced - crispy chicken, crunchy peanuts, "
                "mild spice level. Great texture contrast and not too oily.",
                4.5, "Mike K."
            ),
            ReviewDemo(
                "The dry-fried green beans were amazing! Crispy outside, tender inside. "
                "Light seasoning with garlic and preserved vegetables. Very fresh taste.",
                4.0, "Lisa C."
            ),
            ReviewDemo(
                "Dan dan noodles had an excellent sauce - nutty sesame with mild heat. "
                "Chewy noodles and ground pork. Rich umami flavor but quite salty.",
                4.0, "David L."
            ),
            ReviewDemo(
                "Mapo tofu is seriously spicy! If you're sensitive to heat, ask for mild. "
                "The numbness from peppercorns lasts for minutes. Generous portion size.",
                4.5, "Jenny T."
            ),
            ReviewDemo(
                "Hot and sour soup was disappointing - too thick and gloopy. "
                "Not enough sourness, mostly just black pepper heat. Skip this one.",
                2.0, "Alex R."
            ),
        ]

    def demo_dish_matching(self, dish_name: str) -> List[str]:
        """
        Demo of dish-to-review matching using simple keyword matching.
        In the real app, this uses semantic embeddings.
        """
        print(f"\n🔍 Finding reviews for: '{dish_name}'")
        print("-" * 40)

        # Simple keyword matching (real app uses embeddings)
        dish_keywords = dish_name.lower().split()
        matching_reviews = []

        for review in self.demo_reviews:
            review_text = review.text.lower()

            # Check if any dish keywords appear in review
            if any(keyword in review_text for keyword in dish_keywords):
                matching_reviews.append(review.text)

        if matching_reviews:
            print(f"Found {len(matching_reviews)} matching reviews:")
            for i, review in enumerate(matching_reviews, 1):
                print(f"\n{i}. {review[:100]}...")
        else:
            print("No matching reviews found.")

        return matching_reviews

    def demo_aspect_extraction(self, review_snippets: List[str]) -> Dict[str, Any]:
        """
        Demo of Aspect-Based Sentiment Analysis (ABSA).
        Extracts taste aspects from review text using rules.
        """
        print(f"\n🧠 Extracting taste aspects from {len(review_snippets)} reviews...")
        print("-" * 50)

        aspects = {
            "spice": 0,
            "heat_type": [],
            "texture": [],
            "richness": None,
            "portion": None,
            "allergens": [],
            "tags": []
        }

        for snippet in review_snippets:
            text = snippet.lower()

            # Spice level detection
            if any(word in text for word in ["very spicy", "seriously spicy", "incredible.*spicy"]):
                aspects["spice"] = max(aspects["spice"], 3)
            elif any(word in text for word in ["spicy", "heat"]):
                aspects["spice"] = max(aspects["spice"], 2)
            elif "mild" in text:
                aspects["spice"] = max(aspects["spice"], 1)

            # Heat type detection
            if any(word in text for word in ["numbing", "peppercorn", "sichuan"]):
                if "peppercorn" not in aspects["heat_type"]:
                    aspects["heat_type"].append("peppercorn")
            if "pepper" in text and "black pepper" in text:
                if "black_pepper" not in aspects["heat_type"]:
                    aspects["heat_type"].append("black_pepper")

            # Texture detection
            texture_words = {
                "crispy": "crispy", "crunchy": "crunchy", "silky": "silky",
                "chewy": "chewy", "tender": "tender", "thick": "thick"
            }
            for word, texture in texture_words.items():
                if word in text and texture not in aspects["texture"]:
                    aspects["texture"].append(texture)

            # Richness detection
            if any(word in text for word in ["rich", "heavy", "oily"]):
                aspects["richness"] = "heavy"
            elif any(word in text for word in ["light", "fresh"]):
                aspects["richness"] = "light"

            # Portion detection
            if any(word in text for word in ["generous", "large", "big"]):
                aspects["portion"] = "large"
            elif "small" in text:
                aspects["portion"] = "small"

            # Allergen detection
            if "peanut" in text:
                aspects["allergens"].append("peanut")

            # Authenticity tags
            if "authentic" in text:
                aspects["tags"].append("authentic")

        # Clean up lists
        aspects["heat_type"] = list(set(aspects["heat_type"]))
        aspects["texture"] = list(set(aspects["texture"]))
        aspects["allergens"] = list(set(aspects["allergens"]))
        aspects["tags"] = list(set(aspects["tags"]))

        print("Extracted aspects:")
        for key, value in aspects.items():
            if value:  # Only show non-empty values
                print(f"  {key}: {value}")

        return aspects

    def demo_taste_card_generation(self, dish_name: str, aspects: Dict[str, Any],
                                 review_snippets: List[str]) -> Dict[str, Any]:
        """
        Demo of taste card generation.
        In the real app, this uses LLM (GPT-4) to generate descriptions.
        """
        print(f"\n📝 Generating taste card for: '{dish_name}'")
        print("-" * 45)

        # Rule-based taste description generation (real app uses LLM)
        bullets = []

        # Spice description
        spice_descriptions = {
            0: "No spice",
            1: "Mild heat that most people can handle",
            2: "Medium spice level with noticeable warmth",
            3: "Very spicy - intense heat that may overwhelm sensitive palates"
        }
        spice_level = aspects.get("spice", 0)
        if spice_level > 0:
            bullets.append(spice_descriptions[spice_level])

        # Heat type description
        if "peppercorn" in aspects.get("heat_type", []):
            bullets.append("Features numbing Sichuan peppercorns that create a tingling sensation")

        # Texture description
        textures = aspects.get("texture", [])
        if textures:
            texture_desc = f"Texture: {', '.join(textures)}"
            bullets.append(texture_desc)

        # Richness description
        richness = aspects.get("richness")
        if richness == "heavy":
            bullets.append("Rich and indulgent - a hearty, satisfying dish")
        elif richness == "light":
            bullets.append("Light and fresh - won't leave you feeling heavy")

        # Portion description
        portion = aspects.get("portion")
        if portion == "large":
            bullets.append("Generous portion size - good for sharing")

        # Allergen warnings
        allergens = aspects.get("allergens", [])
        if allergens:
            bullets.append(f"Contains: {', '.join(allergens)}")

        # Authenticity
        if "authentic" in aspects.get("tags", []):
            bullets.append("Authentic preparation that stays true to traditional flavors")

        # Generate pairing suggestion based on aspects
        pairing = self._generate_pairing_suggestion(dish_name, aspects)

        taste_card = {
            "dish_name": dish_name,
            "restaurant": self.demo_restaurant,
            "bullets": bullets,
            "pairing": pairing,
            "confidence": "high" if len(review_snippets) >= 2 else "medium",
            "sources": [f"{len(review_snippets)} customer reviews analyzed"],
            "spice_level": spice_level,
            "aspects": aspects
        }

        print("Generated taste card:")
        for bullet in bullets:
            print(f"  • {bullet}")
        print(f"\nPairing: {pairing}")
        print(f"Confidence: {taste_card['confidence']}")

        return taste_card

    def _generate_pairing_suggestion(self, dish_name: str, aspects: Dict[str, Any]) -> str:
        """Generate pairing suggestions based on taste aspects."""
        suggestions = []

        spice_level = aspects.get("spice", 0)
        if spice_level >= 2:
            suggestions.append("steamed rice to balance the heat")
            suggestions.append("cold beer to cool the palate")

        if aspects.get("richness") == "heavy":
            suggestions.append("light vegetable dish for balance")

        if not suggestions:
            suggestions.append("steamed rice and tea")

        return f"Pairs well with {', '.join(suggestions)}"

    def demo_combo_scoring(self, dishes: List[str]) -> Dict[str, Any]:
        """
        Demo of combo recommendation scoring.
        Real app uses complex balance algorithm with constraints.
        """
        print(f"\n🍽️ Scoring combo: {', '.join(dishes)}")
        print("-" * 50)

        # Get taste cards for each dish
        dish_data = {}
        for dish in dishes:
            reviews = self.demo_dish_matching(dish)
            if reviews:
                aspects = self.demo_aspect_extraction(reviews)
                dish_data[dish] = aspects

        # Simple scoring algorithm
        score = 0
        rationale_points = []

        # Base score from number of dishes
        score += len(dishes) * 10

        # Spice variety bonus
        spice_levels = [data.get("spice", 0) for data in dish_data.values()]
        if len(set(spice_levels)) > 1:
            score += 15
            rationale_points.append("good spice level variety")

        # Texture variety bonus
        all_textures = []
        for data in dish_data.values():
            all_textures.extend(data.get("texture", []))
        unique_textures = set(all_textures)
        if len(unique_textures) >= 3:
            score += 10
            rationale_points.append("diverse textures")

        # Balance penalty for too much spice
        if all(level >= 3 for level in spice_levels if level > 0):
            score -= 20
            rationale_points.append("may be too spicy overall")

        # Coverage bonus for different courses
        dish_types = [self._classify_dish_type(dish) for dish in dishes]
        if len(set(dish_types)) >= 2:
            score += 20
            rationale_points.append("covers multiple course types")

        rationale = f"Balanced combination with {', '.join(rationale_points)}"

        combo_result = {
            "dishes": dishes,
            "score": score,
            "rationale": rationale,
            "dish_data": dish_data,
            "balance_analysis": {
                "spice_range": [min(spice_levels), max(spice_levels)],
                "texture_variety": list(unique_textures),
                "course_coverage": list(set(dish_types))
            }
        }

        print(f"Combo score: {score}")
        print(f"Rationale: {rationale}")

        return combo_result

    def _classify_dish_type(self, dish_name: str) -> str:
        """Classify dish by type for combo analysis."""
        dish_lower = dish_name.lower()
        if any(word in dish_lower for word in ["soup"]):
            return "soup"
        elif any(word in dish_lower for word in ["rice", "noodle"]):
            return "carb"
        elif any(word in dish_lower for word in ["tofu", "chicken", "pork", "beef"]):
            return "protein"
        elif any(word in dish_lower for word in ["bean", "vegetable"]):
            return "vegetable"
        elif any(word in dish_lower for word in ["ice cream", "dessert"]):
            return "dessert"
        else:
            return "other"

    def run_full_demo(self):
        """Run the complete demo showing all functionality."""
        print("🍜 Menu Taste Guide API - Core Algorithm Demo")
        print("=" * 60)
        print(f"Restaurant: {self.demo_restaurant}")
        print(f"Menu items: {len(self.demo_menu)}")
        print(f"Customer reviews: {len(self.demo_reviews)}")

        # Demo 1: Dish matching and taste card generation
        test_dish = "Mapo Tofu"
        reviews = self.demo_dish_matching(test_dish)
        if reviews:
            aspects = self.demo_aspect_extraction(reviews)
            taste_card = self.demo_taste_card_generation(test_dish, aspects, reviews)

        # Demo 2: Combo recommendation
        test_combo = ["Mapo Tofu", "Dry-Fried Green Beans", "Steamed Rice"]
        combo_result = self.demo_combo_scoring(test_combo)

        # Summary
        print("\n" + "=" * 60)
        print("🎉 Demo Complete!")
        print("\nThis demonstrates the core algorithms:")
        print("1. 🔍 Dish-to-Review Matching (semantic similarity in real app)")
        print("2. 🧠 Aspect-Based Sentiment Analysis (ABSA)")
        print("3. 📝 LLM Taste Card Generation (GPT-4 in real app)")
        print("4. 🍽️ Combo Scoring Algorithm")
        print("\nThe real API adds:")
        print("• OCR menu processing (Google Vision)")
        print("• Restaurant lookup (Google Places/Yelp)")
        print("• Semantic embeddings (sentence-transformers)")
        print("• Database persistence (PostgreSQL)")
        print("• Caching layer (Redis)")
        print("• FastAPI web interface")


def main():
    """Run the demo."""
    demo = MenuTasteGuideDemo()
    demo.run_full_demo()


if __name__ == "__main__":
    main()