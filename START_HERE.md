# 🍜 Menu Taste Guide - Quick Start Guide

## ✅ OCR is Now Working!

Your interactive menu app now has **real OCR processing** using Tesseract!

## 🚀 Start the App

```bash
cd /workspace/menu_taste_app
/data/venv/aimenu/bin/python app_with_ocr.py
```

## 🌐 Access the App

Once started, visit:
- **📱 Interactive App:** http://localhost:8000/app
- **📊 API Overview:** http://localhost:8000
- **📚 API Docs:** http://localhost:8000/docs

## 📸 How to Use OCR Feature

### Method 1: Upload Menu Images via Web Interface

1. Go to http://localhost:8000/app
2. Click the **"Menu OCR"** tab
3. Click **"Choose Images"** button
4. Select one or more menu photos from your computer
5. Wait for processing (usually 2-5 seconds)
6. See extracted dishes with prices!

### Method 2: Test with API Directly

```bash
# Upload an image via curl
curl -X POST "http://localhost:8000/api/ocr/menu" \
  -F "images=@/path/to/menu.jpg" \
  -F "restaurant_name=My Restaurant"
```

## 🔍 What the OCR Extracts

From your menu images, it will automatically detect:
- ✅ **Dish names** (e.g., "Kung Pao Chicken")
- ✅ **Prices** (e.g., "$16.95")
- ✅ **Menu sections** (e.g., "Appetizers", "Main Dishes")
- ✅ **Confidence scores** for each item

## 📝 Expected Image Format

**Best results with:**
- Clear, well-lit photos
- High contrast between text and background
- Horizontal text orientation
- Standard fonts (not too stylized)

**Supported formats:**
- JPG/JPEG
- PNG
- BMP
- TIFF

## 🎯 Full Workflow Example

1. **Upload Menu Image** → Extracts dishes automatically
2. **Click "Analyze" on any dish** → Get taste profile
3. **Go to "Combo" tab** → Get balanced meal recommendations
4. **Adjust preferences** → Party size, budget, dietary needs

## 🧪 Test with Demo Data

If you don't have a menu image handy:
1. Click **"Load Demo Menu"** button
2. See sample extracted items instantly
3. Try the taste analysis and combo features

## 🔧 Technical Details

### OCR Engine
- **Tesseract 5.5.1** (open-source OCR engine)
- Installed via conda in `/data/venv/aimenu`
- Configured in `app_with_ocr.py`

### Processing Pipeline
1. **Image Upload** → FastAPI receives the file
2. **OCR Processing** → Tesseract extracts raw text
3. **Text Parsing** → Regex patterns find dishes/prices
4. **Section Detection** → Groups items by category
5. **Deduplication** → Removes similar entries
6. **Response** → Returns structured JSON

### Accuracy Tips
- Multiple images of same menu → Better coverage
- Well-lit photos → Higher OCR accuracy
- Clear fonts → Better text recognition
- Horizontal orientation → Best results

## 🐛 Troubleshooting

### "No menu items detected"
- Try a clearer image
- Ensure good lighting
- Check if text is horizontal
- Try uploading a different page

### "OCR Processing Failed"
- Verify Tesseract is installed: `/data/venv/aimenu/bin/tesseract --version`
- Check image format is supported
- Ensure image file is not corrupted

### Server won't start
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill existing process
pkill -f "python.*app_with_ocr"

# Restart
/data/venv/aimenu/bin/python app_with_ocr.py
```

## 📊 Sample Output

### Input: Menu Image
```
MAIN DISHES
Kung Pao Chicken.......$16.95
Mapo Tofu..............$14.95
Sweet and Sour Pork....$15.95
```

### Output: Extracted Data
```json
{
  "menu_items": [
    {
      "name": "Kung Pao Chicken",
      "price": 16.95,
      "section": "Main Dishes",
      "confidence": 0.85
    },
    {
      "name": "Mapo Tofu",
      "price": 14.95,
      "section": "Main Dishes",
      "confidence": 0.85
    }
  ]
}
```

## 🎉 Next Steps

Once you've extracted menu items:
1. **Analyze Taste** → Click "Analyze" on any dish
2. **Get Recommendations** → Use the Combo tab
3. **Share** → Show friends your meal suggestions!

## 💡 Pro Tips

- **Multiple pages?** Upload them all at once for complete menu
- **Bad extraction?** Try re-taking the photo with better lighting
- **Missing prices?** OCR looks for `$XX.XX` patterns
- **Wrong section?** Algorithm uses keyword matching for categories

---

**Enjoy your AI-powered menu analysis! 🍽️**