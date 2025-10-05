#!/usr/bin/env python3
"""Test Surya OCR for menu text detection."""

from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor

# Load image
print("📖 Loading menu image...")
image = Image.open('menu_example.png')

# Initialize predictors
print("🔧 Loading Surya OCR models...")
foundation_predictor = FoundationPredictor()
det_predictor = DetectionPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor=foundation_predictor)

# Run OCR (recognition will call detection internally)
print("🔍 Running Surya OCR (detection + recognition)...")
rec_results = rec_predictor([image], task_names=["ocr_with_boxes"], det_predictor=det_predictor)
rec_result = rec_results[0]

# Display results
print("="*70)
print(f"📊 Surya OCR detected {len(rec_result.text_lines)} text elements")
print("="*70)

for idx, line in enumerate(rec_result.text_lines):
    text = line.text
    bbox = line.bbox
    conf = line.confidence if hasattr(line, 'confidence') else 0.0

    # Get position (top-left corner)
    x = int(bbox[0])
    y = int(bbox[1])

    print(f"{idx+1:3d}. '{text}' at (x={x}, y={y}) conf={conf:.2f}")

print(f"\n{'='*70}")
print(f"✅ Total detections: {len(rec_result.text_lines)}")
print("="*70)
