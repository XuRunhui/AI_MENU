"""FastAPI app with LLM-based menu classification."""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from typing import List

# OCR imports
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor

# Our new classifiers
from llm_classifier import MenuLineClassifier
from menu_assembler import MenuAssembler

app = FastAPI(title="Menu OCR with LLM Classification")

# Initialize predictors
print("🔧 Initializing Surya OCR...")
foundation_predictor = FoundationPredictor()
det_predictor = DetectionPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor=foundation_predictor)

print("🤖 Initializing LLM Classifier...")
classifier = MenuLineClassifier()

print("🔨 Initializing Menu Assembler...")
assembler = MenuAssembler()

def auto_detect_columns(results):
    """Detect columns using K-means, filtering out standalone prices."""
    from sklearn.cluster import KMeans
    import re

    if not results:
        return [0] * len(results)

    img_width = max([r[0][2][0] for r in results])
    price_only_pattern = re.compile(r'^\s*\$?\s*\d{1,3}(?:[.,]\d{2})?\s*$')

    # Filter out standalone prices for column detection
    non_price_results = []
    non_price_x_coords = []
    non_price_indices = []

    for i, result in enumerate(results):
        text = result[1].strip()
        if not price_only_pattern.match(text):
            non_price_results.append(result)
            non_price_x_coords.append(result[0][0][0])
            non_price_indices.append(i)

    if len(non_price_results) == 0:
        return [0] * len(results)

    x_coords = np.array(non_price_x_coords)

    # Try K-means with different cluster counts
    best_labels = [0] * len(results)

    for n_clusters in [2, 3]:
        if len(x_coords) < n_clusters * 3:
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(x_coords.reshape(-1, 1))
        cluster_centers = sorted(kmeans.cluster_centers_.flatten())

        if n_clusters > 1:
            min_gap = min(cluster_centers[i+1] - cluster_centers[i]
                         for i in range(len(cluster_centers)-1))

            if min_gap > img_width * 0.15:
                # Good separation - assign labels
                x_to_label = {}
                for x, label in zip(non_price_x_coords, labels):
                    x_to_label[x] = label

                # Assign all results
                all_labels = []
                for result in results:
                    x = result[0][0][0]
                    if x in x_to_label:
                        all_labels.append(x_to_label[x])
                    else:
                        # Assign to nearest cluster
                        distances = [abs(x - center) for center in cluster_centers]
                        all_labels.append(distances.index(min(distances)))

                best_labels = all_labels
                print(f"✅ Detected {n_clusters} columns with min gap {min_gap:.0f}px ({min_gap/img_width*100:.1f}% of width)")
                break

    return best_labels

@app.post("/api/ocr/menu")
async def process_menu(images: UploadFile = File(...)):
    """Process menu image with LLM-based classification."""
    try:
        # Load image
        image_data = await images.read()
        image = Image.open(io.BytesIO(image_data))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        print(f"\n{'='*70}")
        print(f"📸 Processing menu image: {image.size[0]}x{image.size[1]}px")
        print(f"{'='*70}\n")

        # Run OCR
        print("🔍 Running Surya OCR...")
        rec_results = rec_predictor([image], task_names=["ocr_with_boxes"], det_predictor=det_predictor)
        rec_result = rec_results[0]

        # Convert to standard format
        ocr_results = []
        for line in rec_result.text_lines:
            bbox = [[line.bbox[0], line.bbox[1]], [line.bbox[2], line.bbox[1]],
                   [line.bbox[2], line.bbox[3]], [line.bbox[0], line.bbox[3]]]
            confidence = getattr(line, 'confidence', 0.9)
            ocr_results.append((bbox, line.text, confidence))

        print(f"✅ Surya OCR detected {len(ocr_results)} text elements\n")

        # Detect columns
        print("📊 Detecting columns...")
        column_assignments = auto_detect_columns(ocr_results)

        # Count items per column
        from collections import Counter
        col_counts = Counter(column_assignments)
        print(f"Column distribution: {dict(col_counts)}\n")

        # Classify lines
        print("🎯 Classifying lines...")
        classifications = classifier.classify_lines(ocr_results, column_assignments)

        # Show classification summary
        from collections import Counter
        label_counts = Counter(c.label for c in classifications)
        print(f"\n📊 Classification results:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")

        # Assemble menu
        print(f"\n🔨 Assembling menu structure...")
        result = assembler.assemble_menu(classifications, ocr_results)

        print(f"\n{'='*70}")
        print(f"✅ COMPLETE: {result['processing_stats']['total_items_detected']} items in "
              f"{len(result['processing_stats']['sections_detected'])} sections")
        print(f"{'='*70}\n")

        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Error: {error_detail}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": error_detail}
        )

@app.get("/")
async def root():
    return {"status": "Menu OCR API with LLM Classification", "version": "2.0"}

if __name__ == "__main__":
    import uvicorn
    print("\n🍜 Menu OCR with LLM Classification")
    print("="*60)
    print("🚀 Starting server...")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8091)
