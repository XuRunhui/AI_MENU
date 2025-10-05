"""Simple but effective menu parser using heuristics + Y-proximity matching."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ParsedItem:
    text: str
    bbox: List[List[float]]
    y_center: float
    label: str  # 'price', 'category', 'dish_candidate', 'description', 'junk'

class SimpleMenuParser:
    """Parse menu using simple but effective heuristics."""

    def __init__(self):
        self.price_pattern = re.compile(r'\$\s*(\d{1,3}(?:[.,]\d{2})?)')
        self.junk_patterns = [
            r'catering', r'order\s+online', r'follow\s+us', r'www\.',
            r'https?://', r'@\w+', r'yelp', r'delivery',
        ]
        self.junk_re = re.compile('|'.join(self.junk_patterns), re.IGNORECASE)

    def classify_line(self, text: str, bbox: List[List[float]]) -> str:
        """Classify a single line using simple rules."""
        text_clean = text.strip()

        # Price? - standalone price text (must have $ or be very short numeric)
        # Match: $5, $12, $ 15, $12.50, etc.
        price_only = re.match(r'^\s*<?[bB/]*>?\s*[\$S]?\s*\$?\s*\d{1,3}(?:[.,]\d{2})?\s*<?[/bB]*>?\s*$', text_clean)
        if price_only and len(text_clean) < 12:
            return 'price'

        # Junk?
        if self.junk_re.search(text_clean):
            return 'junk'

        # Description? (long, has commas, lowercase start, or ingredients keywords)
        ingredient_words = ['eggs', 'cheese', 'tortilla', 'beans', 'sauce', 'served with']
        has_ingredients = any(word in text_clean.lower() for word in ingredient_words)

        if (len(text_clean.split()) >= 8 or
            text_clean.count(',') >= 2 or
            (text_clean and text_clean[0].islower()) or
            (has_ingredients and len(text_clean.split()) >= 5)):
            return 'description'

        # Category? (short, all caps, specific generic section words)
        section_keywords = ['omelette', 'plates', 'wraps', 'salads', 'drinks',
                           'sweets', 'appetizers', 'entrees', 'desserts', 'sides', 'menu']
        if (text_clean.isupper() and
            1 <= len(text_clean.split()) <= 3 and
            any(kw in text_clean.lower() for kw in section_keywords)):
            return 'category'

        # Dish name should NOT look like HTML tags or be very short standalone numbers
        if re.match(r'^<?[/bB]+>?$', text_clean) or re.match(r'^\d{1,2}$', text_clean):
            return 'junk'

        # Everything else is a potential dish name
        return 'dish_candidate'

    def match_dish_to_price(self, dish_y: float, prices: List[ParsedItem],
                           y_threshold: float = 25.0) -> Optional[ParsedItem]:
        """Find the closest price within Y threshold."""
        best_price = None
        min_dist = float('inf')

        for price in prices:
            dist = abs(price.y_center - dish_y)
            if dist < y_threshold and dist < min_dist:
                min_dist = dist
                best_price = price

        return best_price

    def find_descriptions(self, dish_y: float, descriptions: List[ParsedItem],
                         max_gap: float = 60.0) -> List[str]:
        """Find descriptions below the dish."""
        result = []
        for desc in descriptions:
            if dish_y < desc.y_center < dish_y + max_gap:
                result.append(desc.text)
        return result

    def parse_menu(self, ocr_results: List[Tuple]) -> Dict[str, Any]:
        """Parse OCR results into structured menu."""

        # Step 1: Classify all lines
        parsed_items = []
        for bbox, text, conf in ocr_results:
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            label = self.classify_line(text, bbox)

            parsed_items.append(ParsedItem(
                text=text.strip(),
                bbox=bbox,
                y_center=y_center,
                label=label
            ))

        # Separate by type
        prices = [p for p in parsed_items if p.label == 'price']
        categories = [p for p in parsed_items if p.label == 'category']
        dish_candidates = [p for p in parsed_items if p.label == 'dish_candidate']
        descriptions = [p for p in parsed_items if p.label == 'description']

        print(f"📊 Parsed: {len(categories)} categories, {len(dish_candidates)} dish candidates, "
              f"{len(prices)} prices, {len(descriptions)} descriptions")

        # Step 2: Match dishes to prices
        menu_items = []
        current_category = "Other"

        # Sort categories by Y
        categories.sort(key=lambda x: x.y_center)

        # Process dish candidates
        dish_candidates.sort(key=lambda x: x.y_center)

        for dish in dish_candidates:
            # Update category (last category above this dish)
            for cat in categories:
                if cat.y_center < dish.y_center:
                    current_category = cat.text.title()

            # Find price
            price_item = self.match_dish_to_price(dish.y_center, prices)

            # Extract price value
            price_val = None
            price_text = None
            if price_item:
                match = self.price_pattern.search(price_item.text)
                if match:
                    price_str = match.group(1).replace(',', '.')
                    try:
                        price_val = float(price_str)
                        price_text = f"${price_val:.2f}"
                    except:
                        pass

            # Find descriptions
            dish_descs = self.find_descriptions(dish.y_center, descriptions)
            combined_desc = ' '.join(dish_descs) if dish_descs else None

            # Only include if has price or is in a known category
            if price_val or current_category != "Other":
                menu_items.append({
                    'category': current_category,
                    'dish_name': dish.text,
                    'price': price_val,
                    'price_text': price_text,
                    'description': combined_desc,
                    'y_position': dish.y_center
                })

        # Step 3: Build structure
        menu_structure = {}
        for item in menu_items:
            cat = item['category']
            if cat not in menu_structure:
                menu_structure[cat] = []

            menu_structure[cat].append({
                'name': item['dish_name'],
                'price': item['price'],
                'price_text': item['price_text'],
                'description': item['description'],
                'confidence': 0.90,  # Default confidence for heuristic matching
                'order': len(menu_structure[cat])
            })

        # Stats
        total = sum(len(items) for items in menu_structure.values())
        with_prices = sum(1 for cat in menu_structure.values()
                         for item in cat if item['price'])

        print(f"\n✅ Final menu: {total} items in {len(menu_structure)} sections")
        print(f"   Items with prices: {with_prices}/{total}")

        return {
            'menu_structure': menu_structure,
            'processing_stats': {
                'total_items_detected': total,
                'items_with_prices': with_prices,
                'sections_detected': list(menu_structure.keys())
            }
        }
