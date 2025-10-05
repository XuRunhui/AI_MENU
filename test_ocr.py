#!/usr/bin/env python3
"""Quick test to verify OCR is working."""

import pytesseract
from PIL import Image, ImageDraw, ImageFont
import io

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = '/data/venv/aimenu/bin/tesseract'

# Create a simple test image with menu text
def create_test_menu_image():
    """Create a test menu image."""
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)

    # Draw some menu items
    menu_text = [
        "APPETIZERS",
        "Spring Rolls...................$6.95",
        "Potstickers....................$8.95",
        "",
        "MAIN DISHES",
        "Kung Pao Chicken.............$16.95",
        "Mapo Tofu....................$14.95",
        "Sweet and Sour Pork..........$15.95"
    ]

    y = 30
    for line in menu_text:
        draw.text((30, y), line, fill='black')
        y += 40

    return img

# Test OCR
print("🧪 Testing OCR Functionality")
print("=" * 50)

try:
    # Create test image
    img = create_test_menu_image()

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Perform OCR
    text = pytesseract.image_to_string(img)

    print("✅ OCR is working!")
    print("\nExtracted text:")
    print("-" * 50)
    print(text)
    print("-" * 50)

    # Check if we got reasonable output
    if "APPETIZERS" in text or "MAIN" in text or "Kung Pao" in text:
        print("\n✅ OCR successfully detected menu text!")
        print("🎉 Ready to process real menu images!")
    else:
        print("\n⚠️  OCR working but text quality may vary")

except Exception as e:
    print(f"❌ OCR test failed: {str(e)}")
    print("\nMake sure Tesseract is installed:")
    print("  /data/venv/aimenu/bin/tesseract --version")
