#!/usr/bin/env python3
"""Test PaddleOCR v4 for menu text detection."""

import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize PaddleOCR v4 with updated API
print("🔧 Initializing PaddleOCR v4...")
ocr = PaddleOCR(
    use_textline_orientation=True,  # Enable text orientation detection
    lang='en',
    device='cpu'
)

# Load image
print("📖 Loading menu image...")
img = cv2.imread('menu_example.png')

# Run OCR (use predict instead of ocr in v4)
print("🔍 Running PaddleOCR detection...\n")
result = ocr.predict(img)

# Display results - PaddleOCR v4 returns OCRResult object
print("="*70)
print(f"Result type: {type(result)}")
print("="*70)

# Access results from OCRResult object
if result and len(result) > 0:
    ocr_result = result[0]

    # Check available attributes and keys
    print(f"\nOCRResult type: {type(ocr_result)}")
    print(f"OCRResult keys: {ocr_result.keys() if hasattr(ocr_result, 'keys') else 'N/A'}")
    print(f"OCRResult items: {list(ocr_result.items())[:3] if hasattr(ocr_result, 'items') else 'N/A'}")

    # Try to access detections (dict access, not attribute)
    if 'dt_polys' in ocr_result:
        detections = ocr_result['dt_polys']
        texts = ocr_result.get('rec_texts', [])
        scores = ocr_result.get('rec_scores', [])

        print(f"\n📊 PaddleOCR detected {len(detections)} text elements")
        print("="*70)

        for idx in range(len(detections)):
            bbox = detections[idx]
            text = texts[idx] if idx < len(texts) else ''
            conf = scores[idx] if idx < len(scores) else 0.0

            # Get position (top-left corner)
            x = int(bbox[0][0])
            y = int(bbox[0][1])

            print(f"{idx+1:3d}. '{text}' at (x={x}, y={y}) conf={conf:.2f}")

        print(f"\n{'='*70}")
        print(f"✅ Total detections: {len(detections)}")
        print("="*70)
    else:
        print("❌ Could not find detection data in result")
else:
    print("❌ No detections found")
