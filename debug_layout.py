"""Debug layout features for specific lines."""

from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from hybrid_classifier_v2 import ImprovedHybridClassifier
import numpy as np
import re
from sklearn.cluster import KMeans

# Initialize
print("🔧 Initializing...")
foundation_predictor = FoundationPredictor()
det_predictor = DetectionPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor=foundation_predictor)
classifier = ImprovedHybridClassifier()

# Load test image
image = Image.open("menu_example.png")
if image.mode != 'RGB':
    image = image.convert('RGB')

# OCR
print("🔍 Running OCR...")
rec_results = rec_predictor([image], task_names=["ocr_with_boxes"], det_predictor=det_predictor)
rec_result = rec_results[0]

ocr_results = []
for line in rec_result.text_lines:
    bbox = [[line.bbox[0], line.bbox[1]], [line.bbox[2], line.bbox[1]],
           [line.bbox[2], line.bbox[3]], [line.bbox[0], line.bbox[3]]]
    conf = getattr(line, 'confidence', 0.9)
    ocr_results.append((bbox, line.text, conf))

# Column detection
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
        min_gap = min(cluster_centers[i+1] - cluster_centers[i] for i in range(len(cluster_centers)-1))
        if min_gap > img_width * 0.15:
            all_labels = []
            for result in results:
                x = result[0][0][0]
                distances = [abs(x - center) for center in cluster_centers]
                all_labels.append(distances.index(min(distances)))
            return all_labels
    return [0] * len(results)

column_assignments = auto_detect_columns(ocr_results)

# Compute layout features
print("\n📐 Computing layout features...")
layout_features = classifier.compute_layout_features(ocr_results, column_assignments)

# Check the four categories
category_lines = [0, 1, 2, 65]  # SPECIALS, Poutines, Sandwiches, Kids Menu

print(f"\n{'='*80}")
print(f"CATEGORY TEXT SIZE ANALYSIS")
print(f"{'='*80}")

for line_idx in category_lines:
    text = ocr_results[line_idx][1]
    feat = layout_features[line_idx]

    print(f"\nLine {line_idx}: '{text}'")
    print(f"  Height: {feat['height']:.2f} pixels")
    print(f"  Relative Height: {feat['relative_height']:.2f}")
    print(f"  Size Class: {feat['size_class']}")
    print(f"  Indent Level: {feat['indent_level']}")
    print(f"  Gap Class: {feat['gap_class']}")
    print(f"  Alignment: {feat['alignment']}")

# Show median height for reference
heights = [f["height"] for f in layout_features]
median_height = np.median(heights)
print(f"\n{'='*80}")
print(f"Median height across all text: {median_height:.2f} pixels")
print(f"Large threshold (>1.4x): >{median_height * 1.4:.2f} pixels")
print(f"Small threshold (<0.8x): <{median_height * 0.8:.2f} pixels")
print(f"{'='*80}")

# Show distribution
size_counts = {}
for f in layout_features:
    size_counts[f['size_class']] = size_counts.get(f['size_class'], 0) + 1

print(f"\nSize class distribution:")
for size, count in sorted(size_counts.items()):
    print(f"  {size}: {count} items ({count*100//len(layout_features)}%)")
