"""Assemble menu structure from classified OCR lines."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from llm_classifier import LineClassification

@dataclass
class MenuItem:
    """A complete menu item with dish name, price, and description."""
    category: str
    dish_name: str
    price: Optional[float]
    price_text: Optional[str]
    description: Optional[str]
    confidence: float
    y_position: float

class MenuAssembler:
    """Assemble classified lines into structured menu items."""

    def __init__(self):
        self.price_pattern = re.compile(r'\$?\s*(\d{1,3}(?:[.,]\d{2})?)')

    def extract_price_value(self, price_text: str) -> Tuple[Optional[float], str]:
        """Extract numeric price value from text."""
        match = self.price_pattern.search(price_text)
        if match:
            price_str = match.group(1).replace(',', '.')
            try:
                if '.' not in price_str:
                    price_val = float(price_str)
                else:
                    price_val = float(price_str)
                # Format display text
                display = f"${price_val:.2f}"
                return price_val, display
            except ValueError:
                pass
        return None, price_text

    def match_prices_to_dishes(self,
                               classified_lines: List[LineClassification],
                               ocr_results: List[Dict[str, Any]],
                               y_threshold: float = 20.0) -> List[MenuItem]:
        """Match prices to dishes using Y-coordinate proximity."""

        # Separate by label
        dishes = []
        prices = []
        descriptions = []
        categories = []

        for i, cls in enumerate(classified_lines):
            bbox, text, conf = ocr_results[i]
            y_center = (bbox[0][1] + bbox[2][1]) / 2

            item_data = {
                'index': i,
                'text': text,
                'bbox': bbox,
                'y_center': y_center,
                'confidence': conf,
                'classification': cls
            }

            if cls.label == "dish":
                dishes.append(item_data)
            elif cls.label == "price":
                prices.append(item_data)
            elif cls.label == "description":
                descriptions.append(item_data)
            elif cls.label == "category":
                categories.append(item_data)

        print(f"📋 Found: {len(categories)} categories, {len(dishes)} dishes, "
              f"{len(prices)} prices, {len(descriptions)} descriptions")

        # Match dishes to prices by Y-proximity
        menu_items = []
        current_category = "Other"

        # Sort categories by Y position
        categories.sort(key=lambda x: x['y_center'])

        # Process dishes in order
        dishes.sort(key=lambda x: x['y_center'])

        for dish in dishes:
            # Find current category (last category above this dish)
            for cat in categories:
                if cat['y_center'] < dish['y_center']:
                    current_category = cat['text'].strip().title()

            # Find closest price (within y_threshold)
            closest_price = None
            min_distance = float('inf')

            for price in prices:
                y_distance = abs(price['y_center'] - dish['y_center'])
                if y_distance < y_threshold and y_distance < min_distance:
                    min_distance = y_distance
                    closest_price = price

            # Extract price value
            price_val = None
            price_text = None
            if closest_price:
                price_val, price_text = self.extract_price_value(closest_price['text'])

            # Find descriptions (below the dish, close Y proximity)
            dish_descriptions = []
            for desc in descriptions:
                # Description should be below dish and within threshold
                if (desc['y_center'] > dish['y_center'] and
                    desc['y_center'] < dish['y_center'] + 100):  # Within 100px below
                    y_gap = desc['y_center'] - dish['y_center']
                    if y_gap < 50:  # Close proximity
                        dish_descriptions.append(desc['text'])

            combined_description = ' '.join(dish_descriptions) if dish_descriptions else None

            menu_items.append(MenuItem(
                category=current_category,
                dish_name=dish['text'].strip(),
                price=price_val,
                price_text=price_text,
                description=combined_description,
                confidence=dish['confidence'],
                y_position=dish['y_center']
            ))

        return menu_items

    def post_process_smoothing(self, menu_items: List[MenuItem]) -> List[MenuItem]:
        """Apply post-processing rules to improve quality."""
        # Sort by category and Y position
        menu_items.sort(key=lambda x: (x.category, x.y_position))

        # Smooth: if two consecutive items in same category and second has commas,
        # consider merging as description
        smoothed = []
        skip_next = False

        for i, item in enumerate(menu_items):
            if skip_next:
                skip_next = False
                continue

            # Check if next item should be merged as description
            if i < len(menu_items) - 1:
                next_item = menu_items[i + 1]
                if (next_item.category == item.category and
                    not next_item.price and
                    (',' in next_item.dish_name or len(next_item.dish_name.split()) > 6)):
                    # Merge next as description
                    item.description = (item.description + ' ' + next_item.dish_name
                                      if item.description else next_item.dish_name)
                    skip_next = True

            smoothed.append(item)

        return smoothed

    def build_menu_structure(self, menu_items: List[MenuItem]) -> Dict[str, List[Dict[str, Any]]]:
        """Build final menu structure grouped by category."""
        structure = {}

        for item in menu_items:
            if item.category not in structure:
                structure[item.category] = []

            structure[item.category].append({
                "name": item.dish_name,
                "price": item.price,
                "price_text": item.price_text,
                "confidence": round(item.confidence, 2),
                "description": item.description,
                "order": len(structure[item.category])
            })

        return structure

    def assemble_menu(self,
                     classified_lines: List[LineClassification],
                     ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Complete menu assembly pipeline."""

        # Step 1: Match prices to dishes
        menu_items = self.match_prices_to_dishes(classified_lines, ocr_results)

        # Step 2: Post-process smoothing
        menu_items = self.post_process_smoothing(menu_items)

        # Step 3: Build structure
        menu_structure = self.build_menu_structure(menu_items)

        # Calculate stats
        total_items = sum(len(items) for items in menu_structure.values())
        items_with_prices = sum(1 for cat in menu_structure.values()
                               for item in cat if item['price'] is not None)
        avg_conf = (sum(item['confidence'] for cat in menu_structure.values() for item in cat) / total_items
                   if total_items > 0 else 0)

        return {
            "menu_structure": menu_structure,
            "processing_stats": {
                "total_items_detected": total_items,
                "items_with_prices": items_with_prices,
                "average_confidence": round(avg_conf, 2),
                "sections_detected": list(menu_structure.keys())
            }
        }
