"""FastAPI app with hybrid classification approach."""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
from typing import List

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor

from hybrid_classifier_v2 import ImprovedHybridClassifier
from menu_assembler import MenuAssembler

app = FastAPI(
    title="Menu Taste Guide - Hybrid Classification",
    description="AI-powered menu recommendation with Hybrid OCR Classification",
    version="2.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize
print("🔧 Initializing Surya OCR...")
foundation_predictor = FoundationPredictor()
det_predictor = DetectionPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor=foundation_predictor)

print("🤖 Initializing Hybrid Classifier v2 (Heuristics + LLM with Token Budgeting)...")
classifier = ImprovedHybridClassifier()

print("🔨 Initializing Menu Assembler...")
assembler = MenuAssembler()

def auto_detect_columns(results):
    """Quick column detection."""
    from sklearn.cluster import KMeans
    import re

    if not results:
        return [0] * len(results)

    img_width = max([r[0][2][0] for r in results])
    price_only_pattern = re.compile(r'^\s*\$?\s*\d{1,3}(?:[.,]\d{2})?\s*$')

    # Filter prices
    non_price_x = []
    for result in results:
        text = result[1].strip()
        if not price_only_pattern.match(text):
            non_price_x.append(result[0][0][0])

    if len(non_price_x) == 0:
        return [0] * len(results)

    x_coords = np.array(non_price_x)

    # Try K-means
    for n_clusters in [2, 3]:
        if len(x_coords) < n_clusters * 3:
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(x_coords.reshape(-1, 1))
        cluster_centers = sorted(kmeans.cluster_centers_.flatten())

        min_gap = min(cluster_centers[i+1] - cluster_centers[i]
                     for i in range(len(cluster_centers)-1))

        if min_gap > img_width * 0.15:
            # Assign all results
            all_labels = []
            for result in results:
                x = result[0][0][0]
                distances = [abs(x - center) for center in cluster_centers]
                all_labels.append(distances.index(min(distances)))

            print(f"✅ Detected {n_clusters} columns (gap: {min_gap:.0f}px)")
            return all_labels

    return [0] * len(results)

@app.post("/api/ocr/menu")
async def process_menu(images: UploadFile = File(...)):
    """Process menu with hybrid classification."""
    try:
        # Load image
        image_data = await images.read()
        image = Image.open(io.BytesIO(image_data))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        print(f"\n{'='*70}")
        print(f"📸 Processing: {image.size[0]}x{image.size[1]}px")
        print(f"{'='*70}\n")

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

        # Detect columns
        column_assignments = auto_detect_columns(ocr_results)

        # Classify with hybrid approach
        print("🎯 Classifying lines with hybrid approach...")
        classifications = classifier.classify_lines(ocr_results, column_assignments)

        # Assemble menu
        print(f"\n🔨 Assembling menu structure...")
        result = assembler.assemble_menu(classifications, ocr_results)

        print(f"\n{'='*70}")
        print(f"✅ COMPLETE: {result['processing_stats']['total_items_detected']} items")
        print(f"{'='*70}\n")

        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        print(f"❌ Error: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/app", response_class=HTMLResponse)
async def interactive_app():
    """Serve the interactive web interface."""
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint."""
    return """
    <html>
        <head>
            <title>Menu Taste Guide - Hybrid v2</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { color: #2c5aa0; border-bottom: 2px solid #2c5aa0; padding-bottom: 10px; }
                .btn { background: #2c5aa0; color: white; padding: 12px 24px;
                       text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 5px; }
                .btn:hover { background: #1e4080; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🍜 Menu Taste Guide - Hybrid Classification v2</h1>
            </div>
            <h2>Welcome!</h2>
            <p>Your AI-powered menu recommendation system with Hybrid Classification (Heuristics 60% + LLM 40%) is ready.</p>
            <div>
                <a href="/app" class="btn">🖥️ Launch Interactive App</a>
                <a href="/docs" class="btn">📚 API Documentation</a>
            </div>
            <h3>Features:</h3>
            <ul>
                <li>📸 Hybrid OCR Classification (Heuristics + LLM with Token Budgeting)</li>
                <li>🎯 Aggressive heuristics handle 60% of lines</li>
                <li>🤖 LLM (Qwen 2.5-14B) for 40% ambiguous cases</li>
                <li>🔄 Retry mechanism with ultra-short prompts</li>
                <li>🧠 Taste Analysis & Combo Recommendations</li>
            </ul>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Hybrid Menu OCR v2 is running",
        "version": "2.0",
        "features": [
            "Hybrid Classification",
            "Token Budgeting",
            "Retry Mechanism",
            "Markdown Fence Handling"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🍜 Hybrid Menu OCR v2 (Heuristics + LLM with Token Budgeting)")
    print("="*60)
    print("🖥️ Interactive App: http://localhost:8093/app")
    print("📚 API Docs: http://localhost:8093/docs")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8093)
