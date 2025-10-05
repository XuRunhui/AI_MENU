#!/usr/bin/env python3
"""Debug column detection on Mexican menu."""

import cv2
import numpy as np
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from PIL import Image
from sklearn.cluster import KMeans

# Load image
image = Image.open('menu_example1.jpg')

# Initialize predictors
foundation_predictor = FoundationPredictor()
det_predictor = DetectionPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor=foundation_predictor)

# Run OCR
rec_results = rec_predictor([image], task_names=["ocr_with_boxes"], det_predictor=det_predictor)
rec_result = rec_results[0]

# Convert to EasyOCR format
results = []
for line in rec_result.text_lines:
    bbox = [[line.bbox[0], line.bbox[1]], [line.bbox[2], line.bbox[1]],
            [line.bbox[2], line.bbox[3]], [line.bbox[0], line.bbox[3]]]
    results.append((bbox, line.text, line.confidence if hasattr(line, 'confidence') else 0.9))

print(f"Total OCR detections: {len(results)}")

# Get X coordinates
x_coords = np.array([r[0][0][0] for r in results])
img_width = max([r[0][2][0] for r in results])

print(f"Image width: {img_width}")
print(f"X coordinate range: {x_coords.min()} - {x_coords.max()}")

# Try K-means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(x_coords.reshape(-1, 1))

cluster_centers = sorted(kmeans.cluster_centers_.flatten())
print(f"\nK-means cluster centers: {cluster_centers}")
print(f"Gap between clusters: {cluster_centers[1] - cluster_centers[0]}")
print(f"Gap as % of image width: {(cluster_centers[1] - cluster_centers[0]) / img_width * 100:.1f}%")

# Count items in each cluster
unique, counts = np.unique(labels, return_counts=True)
print(f"\nCluster sizes: {dict(zip(unique, counts))}")

# Show some samples from each cluster
print("\n" + "="*70)
print("Sample items from each cluster:")
print("="*70)
for cluster_id in unique:
    print(f"\nCluster {cluster_id} (center at x={cluster_centers[cluster_id]:.0f}):")
    cluster_items = [(r[1], r[0][0][0]) for r, l in zip(results, labels) if l == cluster_id]
    for text, x in sorted(cluster_items, key=lambda item: item[1])[:10]:
        print(f"  x={x:4.0f}: '{text}'")
