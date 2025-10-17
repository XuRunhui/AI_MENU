"""Show what the LLM actually receives."""

from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from hybrid_classifier_v2 import ImprovedHybridClassifier
import numpy as np
import re
import json
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
layout_features = classifier.compute_layout_features(ocr_results, column_assignments)

# Simulate classification process
ambiguous = []
for i, (result, col_idx) in enumerate(zip(ocr_results, column_assignments)):
    bbox, text, conf = result
    line_id = f"line_{i}"

    heuristic_result = classifier.aggressive_heuristic(text, bbox, col_idx, conf)

    if not heuristic_result:
        # This goes to LLM
        text_clean = re.sub(r'<[^>]+>', '', text).strip()
        layout_feat = layout_features[i]

        ambiguous.append({
            "id": line_id,
            "text": text_clean,
            "size_class": layout_feat["size_class"],
            "indent_level": layout_feat["indent_level"],
            "gap_class": layout_feat["gap_class"],
            "alignment": layout_feat["alignment"],
        })

print(f"\n{'='*80}")
print(f"ITEMS SENT TO LLM (with layout features)")
print(f"{'='*80}\n")

for item in ambiguous[:15]:  # Show first 15
    print(json.dumps(item, indent=2))
    print()

print(f"\nTotal items sent to LLM: {len(ambiguous)}/98")
