"""Test the improved layout-aware classifier."""

from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from hybrid_classifier_v2 import ImprovedHybridClassifier
import numpy as np

# Initialize
print("🔧 Initializing Surya OCR...")
foundation_predictor = FoundationPredictor()
det_predictor = DetectionPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor=foundation_predictor)

print("🤖 Initializing Hybrid Classifier with Layout Features...")
classifier = ImprovedHybridClassifier()

# Load test image
print("\n📸 Loading menu_example.png...")
image = Image.open("menu_example.png")
if image.mode != 'RGB':
    image = image.convert('RGB')

# OCR
print("🔍 Running Surya OCR...")
rec_results = rec_predictor([image], task_names=["ocr_with_boxes"], det_predictor=det_predictor)
rec_result = rec_results[0]

# Convert format
ocr_results = []
for line in rec_result.text_lines:
    bbox = [[line.bbox[0], line.bbox[1]], [line.bbox[2], line.bbox[1]],
           [line.bbox[2], line.bbox[3]], [line.bbox[0], line.bbox[3]]]
    conf = getattr(line, 'confidence', 0.9)
    ocr_results.append((bbox, line.text, conf))

print(f"✅ Detected {len(ocr_results)} text elements\n")

# Simple column detection
from sklearn.cluster import KMeans

def auto_detect_columns(results):
    if not results:
        return [0] * len(results)

    img_width = max([r[0][2][0] for r in results])
    price_only_pattern = re.compile(r'^\s*\$?\s*\d{1,3}(?:[.,]\d{2})?\s*$')

    non_price_x = []
    for result in results:
        text = result[1].strip()
        if not price_only_pattern.match(text):
            non_price_x.append(result[0][0][0])

    if len(non_price_x) == 0:
        return [0] * len(results)

    x_coords = np.array(non_price_x)

    for n_clusters in [2, 3]:
        if len(x_coords) < n_clusters * 3:
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(x_coords.reshape(-1, 1))
        cluster_centers = sorted(kmeans.cluster_centers_.flatten())

        min_gap = min(cluster_centers[i+1] - cluster_centers[i]
                     for i in range(len(cluster_centers)-1))

        if min_gap > img_width * 0.15:
            all_labels = []
            for result in results:
                x = result[0][0][0]
                distances = [abs(x - center) for center in cluster_centers]
                all_labels.append(distances.index(min(distances)))

            print(f"✅ Detected {n_clusters} columns (gap: {min_gap:.0f}px)")
            return all_labels

    return [0] * len(results)

import re
column_assignments = auto_detect_columns(ocr_results)

# Classify with improved classifier
print("\n🎯 Classifying with Layout-Aware Hybrid Classifier...")
classifications = classifier.classify_lines(ocr_results, column_assignments)

# Count improvements
from collections import Counter
label_counts = Counter(c.label for c in classifications)

print(f"\n{'='*70}")
print(f"📊 FINAL STATISTICS")
print(f"{'='*70}")
print(f"Total lines: {len(classifications)}")
for label, count in sorted(label_counts.items()):
    print(f"  {label:12s}: {count:3d} ({count*100//len(classifications):2d}%)")

# Check specific improvements
junk_items = [c for c in classifications if c.label == "junk"]
print(f"\n✅ Junk items detected: {len(junk_items)}")
print("   (Should now include: 'd', 'Z', 'O', 'CH', 'SOLD OUT', 'Leave us a review!', etc.)")

category_items = [c for c in classifications if c.label == "category"]
print(f"\n✅ Categories detected: {len(category_items)}")
for c in category_items:
    idx = int(c.id.split('_')[1])
    text = ocr_results[idx][1]
    print(f"   - {text} ({c.method})")

print(f"\n{'='*70}")
print("✅ Test complete! Check the detailed output above.")
print(f"{'='*70}\n")
