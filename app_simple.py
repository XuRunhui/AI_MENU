"""FastAPI app with simple heuristic-based menu parsing."""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor

from simple_menu_parser import SimpleMenuParser

app = FastAPI(title="Simple Menu OCR")

# Initialize
print("🔧 Initializing Surya OCR...")
foundation_predictor = FoundationPredictor()
det_predictor = DetectionPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor=foundation_predictor)

parser = SimpleMenuParser()

@app.post("/api/ocr/menu")
async def process_menu(images: UploadFile = File(...)):
    """Process menu with simple heuristic parsing."""
    try:
        # Load image
        image_data = await images.read()
        image = Image.open(io.BytesIO(image_data))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        print(f"\n{'='*70}")
        print(f"📸 Processing: {image.size[0]}x{image.size[1]}px")
        print(f"{'='*70}")

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

        # Parse
        result = parser.parse_menu(ocr_results)

        print(f"{'='*70}\n")
        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        print(f"❌ Error: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"status": "Simple Menu OCR", "version": "1.0"}

@app.get("/app")
async def serve_app():
    """Serve the frontend app."""
    from fastapi.responses import FileResponse
    import os
    static_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path)
    return {"error": "Frontend not found", "hint": "The original frontend is configured for port 8090"}

if __name__ == "__main__":
    import uvicorn
    print("\n🍜 Simple Menu OCR")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8092)
