#!/usr/bin/env python3
"""
Enhanced API with actual OCR processing using Surya OCR, PaddleOCR v4, and EasyOCR.
Uses deep learning-based OCR for better accuracy.
"""

import os
import re
import io
from typing import List, Dict, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import numpy as np
import cv2
import uvicorn
from sklearn.cluster import DBSCAN
from paddleocr import PaddleOCR
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import simple parser
from simple_menu_parser import SimpleMenuParser

app = FastAPI(
    title="Menu Taste Guide - Interactive Demo",
    description="AI-powered menu recommendation with OCR",
    version="1.0.0"
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

# Initialize OCR engines (only once for performance)
# Option 1: Surya OCR (fastest, modern, excellent accuracy)
print("🔧 Initializing Surya OCR...")
surya_foundation = FoundationPredictor()
surya_det = DetectionPredictor()
surya_rec = RecognitionPredictor(foundation_predictor=surya_foundation)

# Option 2: PaddleOCR v4 (good accuracy)
print("🔧 Initializing PaddleOCR v4...")
paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en', device='cpu')

# Option 3: EasyOCR (fallback)
print("🔧 Initializing EasyOCR...")
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# OCR engine selection: "surya", "paddle", or "easyocr"
OCR_ENGINE = "surya"  # Default to Surya OCR


def is_noise_text(text: str) -> bool:
    """
    Filter out noise from OCR results (cracks, stray symbols, etc.)

    Args:
        text: Detected text string

    Returns:
        True if text is likely noise, False if valid
    """
    # Strip whitespace
    text = text.strip()

    # Too short (single character noise)
    if len(text) <= 1:
        return True

    # Only punctuation or special characters
    if re.match(r'^[^\w\s]+$', text):
        return True

    # Common OCR noise patterns
    noise_patterns = [
        r'^[:\'\"\`,\.\-_]+$',  # Only punctuation
        r'^[|Il1]+$',            # Vertical lines detected as letters
        r'^\s+$',                # Only whitespace
    ]

    for pattern in noise_patterns:
        if re.match(pattern, text):
            return True

    return False


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing noise and normalizing.

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text
    """
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing dots from cracks
    text = re.sub(r'^[\.]+|[\.]+$', '', text)

    # Common OCR corrections
    corrections = {
        r'\b0(?=\d)': 'O',  # 0 -> O in words (like "0NION" -> "ONION")
        r'\bl(?=[A-Z])': 'I',  # lowercase l -> I before capitals
        r'\brn\b': 'm',  # rn -> m (common error)
        r'\bvv': 'w',  # vv -> w
        r'(?<=[A-Z])l(?=[a-z])': 'i',  # Cl -> Ci
    }

    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)

    # Strip whitespace
    text = text.strip()

    return text


def auto_orient(img_np: np.ndarray) -> np.ndarray:
    """
    Auto-detect and correct image orientation/skew.

    Args:
        img_np: Input image as numpy array

    Returns:
        Deskewed image
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return img_np

    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = rect[-1]

    # Correct angle
    if angle < -45:
        angle += 90

    # Only rotate if angle is significant (> 0.5 degrees)
    if abs(angle) > 0.5:
        h, w = img_np.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return img_np


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image to improve OCR accuracy.

    Args:
        image: PIL Image

    Returns:
        Preprocessed numpy array
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Moderate contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)

    # Moderate sharpness enhancement
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)

    # Convert to numpy array
    img_np = np.array(image)

    # Auto-orient and deskew (only if needed)
    img_np = auto_orient(img_np)

    # Light denoising
    denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 6, 6, 7, 21)

    print("✨ Image preprocessed: orientation+contrast+denoising applied")
    return denoised


def extract_menu_items_from_image(image_bytes: bytes) -> Dict[str, List[dict]]:
    """
    Extract menu items from image using EasyOCR, preserving original order.

    Args:
        image_bytes: Raw image data

    Returns:
        Dictionary with categories as keys and ordered lists of items as values
    """
    # Open image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB and numpy array - skip heavy preprocessing for good quality images
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_np = np.array(image)

    print("✨ Image loaded (preprocessing disabled for this image quality)")

    # Perform OCR with selected engine
    if OCR_ENGINE == "surya":
        print("🔍 Using Surya OCR for text detection...")
        # Surya OCR returns OCRResult with text_lines
        surya_results = surya_rec([image], task_names=["ocr_with_boxes"], det_predictor=surya_det)
        surya_result = surya_results[0]

        # Convert to common format: (bbox, text, confidence)
        results = []
        for line in surya_result.text_lines:
            # Clean HTML tags from text
            text = re.sub(r'<[^>]+>', '', line.text)
            bbox = [[line.bbox[0], line.bbox[1]],
                   [line.bbox[2], line.bbox[1]],
                   [line.bbox[2], line.bbox[3]],
                   [line.bbox[0], line.bbox[3]]]
            conf = line.confidence if hasattr(line, 'confidence') else 0.95
            results.append((bbox, text, conf))

        print(f"✅ Surya OCR detected {len(results)} text elements")

    elif OCR_ENGINE == "paddle":
        print("🔍 Using PaddleOCR v4 for text detection...")
        # PaddleOCR v4 returns OCRResult object
        paddle_result = paddle_ocr.predict(image_np)

        if paddle_result and len(paddle_result) > 0:
            ocr_result = paddle_result[0]
            detections = ocr_result.get('dt_polys', [])
            texts = ocr_result.get('rec_texts', [])
            scores = ocr_result.get('rec_scores', [])

            # Convert to common format: (bbox, text, confidence)
            results = []
            for idx in range(len(detections)):
                bbox = detections[idx]
                text = texts[idx] if idx < len(texts) else ''
                conf = scores[idx] if idx < len(scores) else 0.0
                results.append((bbox, text, conf))
        else:
            results = []

        print(f"✅ PaddleOCR detected {len(results)} text elements")

    else:  # easyocr
        print("🔍 Using EasyOCR for text detection...")
        # EasyOCR returns list of (bbox, text, confidence)
        results = easyocr_reader.readtext(image_np)
        print(f"✅ EasyOCR detected {len(results)} text elements")

    # Debug logging
    print(f"\n{'='*60}")
    print(f"📊 Total OCR detections: {len(results)}")
    print(f"{'='*60}\n")

    # Use simple parser instead of complex row-based approach
    parser = SimpleMenuParser()
    parsed_result = parser.parse_menu(results)

    return parsed_result['menu_structure']


@app.post("/api/ocr/menu")
async def process_menu_ocr(
    images: List[UploadFile] = File(...),
    restaurant_name: str = Form(None),
    city: str = Form(None)
):
    """
    Process uploaded menu images using OCR.

    Args:
        images: List of uploaded image files
        restaurant_name: Optional restaurant name
        city: Optional city

    Returns:
        Extracted menu structure organized by categories in original order
    """
    all_menu_structure = {}

    for image_file in images:
        # Read image data
        image_data = await image_file.read()

        # Extract menu items (returns dict with categories)
        menu_structure = extract_menu_items_from_image(image_data)

        # Merge into all_menu_structure
        for category, items in menu_structure.items():
            if category not in all_menu_structure:
                all_menu_structure[category] = []
            all_menu_structure[category].extend(items)

    # Calculate statistics
    total_items = sum(len(items) for items in all_menu_structure.values())
    items_with_prices = sum(
        1 for items in all_menu_structure.values()
        for item in items if item.get('price') is not None
    )

    # Calculate average confidence
    all_confidences = [
        item['confidence']
        for items in all_menu_structure.values()
        for item in items
    ]
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    return {
        "menu_structure": all_menu_structure,
        "processing_stats": {
            "total_items_detected": total_items,
            "items_with_prices": items_with_prices,
            "average_confidence": round(avg_confidence, 2),
            "sections_detected": list(all_menu_structure.keys()),
            "images_processed": len(images)
        }
    }


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
            <title>Menu Taste Guide</title>
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
                <h1>🍜 Menu Taste Guide - Interactive Demo</h1>
            </div>
            <h2>Welcome!</h2>
            <p>Your AI-powered menu recommendation system is ready.</p>
            <div>
                <a href="/app" class="btn">🖥️ Launch Interactive App</a>
                <a href="/docs" class="btn">📚 API Documentation</a>
            </div>
            <h3>Features:</h3>
            <ul>
                <li>📸 OCR Menu Scanning (Upload images to extract dishes)</li>
                <li>🧠 Taste Analysis (AI-powered taste descriptions)</li>
                <li>🍽️ Combo Recommendations (Balanced meal suggestions)</li>
                <li>🔍 Restaurant Search (Find and analyze restaurants)</li>
            </ul>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Menu Taste Guide API with OCR is running",
        "ocr_available": True,
        "ocr_engine": "EasyOCR (Deep Learning)",
        "features_ready": [
            "OCR Processing (EasyOCR)",
            "Restaurant Lookup",
            "Taste Analysis",
            "Combo Recommendations"
        ]
    }


if __name__ == "__main__":
    print("🍜 Menu Taste Guide - OCR Demo")
    print("=" * 60)
    print("📸 OCR Processing: Enabled (using EasyOCR)")
    print("🖥️ Interactive App: http://localhost:8000/app")
    print("📚 API Docs: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")