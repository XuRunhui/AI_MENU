"""OCR endpoints for menu processing."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.db.base import get_db
from app.schemas.menu import OCRMenuResponse
from app.services.ocr_service import OCRService
from app.services.menu_service import MenuService

router = APIRouter()


@router.post("/menu", response_model=OCRMenuResponse)
async def process_menu_images(
    images: List[UploadFile] = File(..., description="Menu images to process"),
    restaurant_name: str = Form(None, description="Restaurant name (optional)"),
    city: str = Form(None, description="City location (optional)"),
    db: AsyncSession = Depends(get_db)
) -> OCRMenuResponse:
    """
    Process menu images using OCR to extract dish names and prices.

    This endpoint:
    1. Processes uploaded menu images using Google Vision OCR
    2. Extracts dish names, prices, and menu sections
    3. Normalizes and deduplicates similar dish names
    4. Returns structured menu data with bounding boxes

    Args:
        images: List of menu image files (JPG, PNG, HEIC)
        restaurant_name: Optional restaurant name for context
        city: Optional city location for context
        db: Database session

    Returns:
        OCRMenuResponse with extracted menu items and processing stats

    Example:
        ```python
        # Upload 2 menu images
        files = [("images", open("page1.jpg", "rb")), ("images", open("page2.jpg", "rb"))]
        response = requests.post("/api/v1/ocr/menu", files=files)

        # Response includes extracted dishes with prices and locations
        {
            "menu_items": [
                {
                    "name": "Kung Pao Chicken",
                    "section": "Main Dishes",
                    "price_text": "$16.95",
                    "price_amount": 16.95,
                    "bbox": {"x": 120, "y": 340, "width": 200, "height": 25},
                    "confidence_score": 0.89
                }
            ],
            "processing_stats": {
                "total_items_detected": 25,
                "items_with_prices": 23,
                "average_confidence": 0.87
            }
        }
        ```
    """
    try:
        logger.info(f"Processing {len(images)} menu images")

        # Initialize services
        ocr_service = OCRService()
        menu_service = MenuService(db)

        # Validate image files
        for image in images:
            if not image.content_type or not image.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {image.filename} is not a valid image"
                )

        # Process images with OCR
        extracted_items = await ocr_service.process_menu_images(
            images=images,
            restaurant_context={
                "name": restaurant_name,
                "city": city
            }
        )

        logger.info(f"OCR extracted {len(extracted_items)} menu items")

        # Process and normalize menu items
        processed_items = await menu_service.process_extracted_items(
            extracted_items,
            restaurant_name=restaurant_name,
            city=city
        )

        # Generate processing statistics
        processing_stats = {
            "total_items_detected": len(processed_items),
            "items_with_prices": len([item for item in processed_items if item.price_amount]),
            "average_confidence": sum(item.confidence_score or 0 for item in processed_items) / len(processed_items) if processed_items else 0,
            "sections_detected": list(set(item.section for item in processed_items if item.section)),
            "processing_time_ms": 0  # Would be filled by actual timing
        }

        return OCRMenuResponse(
            menu_items=processed_items,
            processing_stats=processing_stats
        )

    except Exception as e:
        logger.error(f"Error processing menu images: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process menu images: {str(e)}"
        )


@router.get("/health")
async def ocr_health_check():
    """Health check for OCR service."""
    try:
        ocr_service = OCRService()
        is_healthy = await ocr_service.health_check()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "OCR",
            "google_vision_available": is_healthy
        }
    except Exception as e:
        logger.error(f"OCR health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "OCR",
            "error": str(e)
        }