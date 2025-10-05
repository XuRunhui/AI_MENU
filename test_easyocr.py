#!/usr/bin/env python3
"""Test the improved EasyOCR implementation."""

import json
from app_with_ocr import extract_menu_items_from_image

# Read the menu image
with open('menu_example.png', 'rb') as f:
    image_bytes = f.read()

print("🔍 Testing EasyOCR on menu_example.png...")
print("=" * 70)

# Extract menu items
menu_structure = extract_menu_items_from_image(image_bytes)

# Display results
print(f"\n📊 Found {len(menu_structure)} categories:\n")

for category, items in menu_structure.items():
    print(f"\n{'='*70}")
    print(f"📁 {category.upper()} ({len(items)} items)")
    print(f"{'='*70}")

    for item in items:
        name = item['name']
        price = item.get('price_text') or 'N/A'
        confidence = item.get('confidence', 0)
        description = item.get('description')
        print(f"  • {name:50} {price:>10}  (conf: {confidence:.2f})")
        if description:
            print(f"     └─ {description}")

# Statistics
total_items = sum(len(items) for items in menu_structure.values())
items_with_prices = sum(
    1 for items in menu_structure.values()
    for item in items if item.get('price') is not None
)

print("\n" + "="*70)
print("📈 STATISTICS")
print("="*70)
print(f"Total items detected: {total_items}")
print(f"Items with prices: {items_with_prices}")
print(f"Categories: {', '.join(menu_structure.keys())}")

# Save to JSON for inspection
with open('menu_ocr_result.json', 'w') as f:
    json.dump(menu_structure, f, indent=2)

print(f"\n💾 Full results saved to: menu_ocr_result.json")
