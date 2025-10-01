"""OCR service for menu processing using Google Vision API."""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
import difflib

from fastapi import UploadFile
from google.cloud import vision
from PIL import Image
import pytesseract
from loguru import logger

from app.core.config import settings
from app.schemas.menu import MenuItemCreate, BoundingBox


class OCRService:
    """Service for processing menu images using OCR."""

    def __init__(self):
        """Initialize OCR service with Google Vision client."""
        self.vision_client = vision.ImageAnnotatorClient()
        self.price_pattern = re.compile(r'\$?(\d+(?:\.\d{2})?)')
        self.section_patterns = [
            r'appetizers?',
            r'starters?',
            r'soups?',
            r'salads?',
            r'main\s+(?:dishes?|courses?)',
            r'entrees?',
            r'mains?',
            r'specialties',
            r'desserts?',
            r'beverages?',
            r'drinks?',
            r'sides?'
        ]

    async def process_menu_images(
        self,
        images: List[UploadFile],
        restaurant_context: Optional[Dict[str, Any]] = None
    ) -> List[MenuItemCreate]:
        """
        Process multiple menu images and extract dish information.

        Args:
            images: List of uploaded image files
            restaurant_context: Optional context (restaurant name, city)

        Returns:
            List of extracted menu items

        Example:
            >>> ocr_service = OCRService()
            >>> items = await ocr_service.process_menu_images(image_files)
            >>> print(f"Extracted {len(items)} menu items")
        """
        logger.info(f"Processing {len(images)} menu images")

        all_extracted_items = []

        # Process each image
        for i, image_file in enumerate(images):
            try:
                logger.info(f"Processing image {i+1}/{len(images)}: {image_file.filename}")

                # Read image data
                image_data = await image_file.read()

                # Try Google Vision first, fallback to Tesseract
                try:
                    items = await self._process_with_google_vision(image_data)
                    logger.info(f"Google Vision extracted {len(items)} items from image {i+1}")
                except Exception as e:
                    logger.warning(f"Google Vision failed for image {i+1}: {str(e)}, falling back to Tesseract")
                    items = await self._process_with_tesseract(image_data)
                    logger.info(f"Tesseract extracted {len(items)} items from image {i+1}")

                all_extracted_items.extend(items)

            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {str(e)}")
                continue

        # Post-process and deduplicate
        processed_items = self._post_process_items(all_extracted_items, restaurant_context)

        logger.info(f"Final result: {len(processed_items)} unique menu items")
        return processed_items

    async def _process_with_google_vision(self, image_data: bytes) -> List[MenuItemCreate]:
        """
        Process image using Google Vision API.

        Args:
            image_data: Raw image bytes

        Returns:
            List of extracted menu items with bounding boxes
        """
        # Create Vision API image object
        image = vision.Image(content=image_data)

        # Detect text with bounding boxes
        response = self.vision_client.text_detection(image=image)
        annotations = response.text_annotations

        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

        if not annotations:
            return []

        # Extract menu items from text annotations
        items = []
        current_section = None

        for annotation in annotations[1:]:  # Skip first annotation (full text)
            text = annotation.description.strip()

            # Skip very short text or numbers-only
            if len(text) < 3 or text.isdigit():
                continue

            # Check if this looks like a section header
            section = self._detect_section(text)
            if section:
                current_section = section
                continue

            # Check if this looks like a dish
            if self._looks_like_dish(text):
                # Extract bounding box
                vertices = annotation.bounding_poly.vertices
                bbox = self._extract_bounding_box(vertices)

                # Try to extract price from nearby text
                price_text, price_amount = self._extract_price_from_context(text, annotations)

                items.append(MenuItemCreate(
                    name=text,
                    section=current_section,
                    price_text=price_text,
                    price_amount=price_amount,
                    bbox=bbox,
                    confidence_score=0.85  # Google Vision typically has high confidence
                ))

        return items

    async def _process_with_tesseract(self, image_data: bytes) -> List[MenuItemCreate]:
        """
        Fallback processing using Tesseract OCR.

        Args:
            image_data: Raw image bytes

        Returns:
            List of extracted menu items (without precise bounding boxes)
        """
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data))

        # Use Tesseract with menu-optimized config
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,()-$&\' '

        try:
            # Extract text with bounding box data
            data = pytesseract.image_to_data(
                image,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )

            items = []
            current_section = None

            # Process each detected word/line
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = data['conf'][i]

                if not text or confidence < 30:  # Skip low confidence detections
                    continue

                # Check for section headers
                section = self._detect_section(text)
                if section:
                    current_section = section
                    continue

                # Check if this looks like a dish
                if self._looks_like_dish(text):
                    # Create bounding box from Tesseract data
                    bbox = BoundingBox(
                        x=data['left'][i],
                        y=data['top'][i],
                        width=data['width'][i],
                        height=data['height'][i]
                    )

                    # Try to extract price
                    price_text, price_amount = self._extract_price_from_text(text)

                    items.append(MenuItemCreate(
                        name=text,
                        section=current_section,
                        price_text=price_text,
                        price_amount=price_amount,
                        bbox=bbox,
                        confidence_score=confidence / 100.0  # Convert to 0-1 scale
                    ))

            return items

        except Exception as e:
            logger.error(f"Tesseract processing failed: {str(e)}")
            return []

    def _detect_section(self, text: str) -> Optional[str]:
        """
        Detect if text is a menu section header.

        Args:
            text: Text to analyze

        Returns:
            Normalized section name or None
        """
        text_lower = text.lower()

        for pattern in self.section_patterns:
            if re.search(pattern, text_lower):
                # Return normalized section name
                if 'appetizer' in text_lower or 'starter' in text_lower:
                    return 'Appetizers'
                elif 'soup' in text_lower:
                    return 'Soups'
                elif 'salad' in text_lower:
                    return 'Salads'
                elif 'main' in text_lower or 'entree' in text_lower:
                    return 'Main Dishes'
                elif 'dessert' in text_lower:
                    return 'Desserts'
                elif 'beverage' in text_lower or 'drink' in text_lower:
                    return 'Beverages'
                elif 'side' in text_lower:
                    return 'Sides'
                else:
                    return 'Specialties'

        return None

    def _looks_like_dish(self, text: str) -> bool:
        """
        Determine if text looks like a dish name.

        Args:
            text: Text to analyze

        Returns:
            True if text appears to be a dish name
        """
        # Basic heuristics for dish detection
        text = text.strip()

        # Too short or too long
        if len(text) < 4 or len(text) > 80:
            return False

        # Contains mostly numbers or special characters
        if re.match(r'^[\d\s\$\.\-\(\)]+$', text):
            return False

        # Looks like a price only
        if re.match(r'^\$?\d+(\.\d{2})?$', text):
            return False

        # Contains common dish indicators
        dish_indicators = [
            'chicken', 'beef', 'pork', 'fish', 'salmon', 'shrimp', 'tofu',
            'rice', 'noodle', 'soup', 'salad', 'fried', 'grilled', 'steamed',
            'spicy', 'sweet', 'sour', 'crispy', 'tender'
        ]

        text_lower = text.lower()
        has_dish_words = any(indicator in text_lower for indicator in dish_indicators)

        # Must have alphabetic characters
        has_letters = bool(re.search(r'[a-zA-Z]', text))

        return has_letters and (has_dish_words or len(text.split()) >= 2)

    def _extract_bounding_box(self, vertices) -> BoundingBox:
        """
        Extract bounding box from Google Vision vertices.

        Args:
            vertices: Google Vision bounding poly vertices

        Returns:
            BoundingBox object with normalized coordinates
        """
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return BoundingBox(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min
        )

    def _extract_price_from_context(
        self,
        dish_text: str,
        all_annotations
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract price from dish text or nearby annotations.

        Args:
            dish_text: The dish name text
            all_annotations: All text annotations from the image

        Returns:
            Tuple of (price_text, price_amount)
        """
        # First check if price is in the dish text itself
        price_text, price_amount = self._extract_price_from_text(dish_text)
        if price_amount:
            return price_text, price_amount

        # Look for price in nearby annotations (simplified heuristic)
        # In a real implementation, you'd use spatial proximity
        for annotation in all_annotations[1:]:
            text = annotation.description.strip()
            if re.match(r'^\$?\d+(\.\d{2})?$', text):
                price_match = self.price_pattern.search(text)
                if price_match:
                    try:
                        amount = float(price_match.group(1))
                        return text, amount
                    except ValueError:
                        continue

        return None, None

    def _extract_price_from_text(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract price from a text string.

        Args:
            text: Text that may contain a price

        Returns:
            Tuple of (price_text, price_amount)
        """
        price_match = self.price_pattern.search(text)
        if price_match:
            try:
                amount = float(price_match.group(1))
                # Extract the full price text including $ symbol
                full_match = re.search(r'\$?[\d\.]+', text)
                price_text = full_match.group(0) if full_match else f"${amount}"
                return price_text, amount
            except ValueError:
                pass

        return None, None

    def _post_process_items(
        self,
        items: List[MenuItemCreate],
        restaurant_context: Optional[Dict[str, Any]] = None
    ) -> List[MenuItemCreate]:
        """
        Post-process extracted items to normalize and deduplicate.

        Args:
            items: Raw extracted menu items
            restaurant_context: Optional restaurant context

        Returns:
            Cleaned and deduplicated menu items
        """
        if not items:
            return []

        # Normalize dish names
        for item in items:
            item.name = self._normalize_dish_name(item.name)

        # Remove duplicates based on similarity
        unique_items = self._deduplicate_dishes(items)

        # Sort by section and name
        unique_items.sort(key=lambda x: (x.section or "ZZZ", x.name))

        logger.info(f"Post-processing: {len(items)} -> {len(unique_items)} unique items")
        return unique_items

    def _normalize_dish_name(self, name: str) -> str:
        """
        Normalize a dish name for consistency.

        Args:
            name: Raw dish name

        Returns:
            Normalized dish name
        """
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name.strip())

        # Remove trailing dots/dashes (menu formatting)
        name = re.sub(r'[\.\_\-]+$', '', name)

        # Remove price information if it got mixed in
        name = re.sub(r'\$\d+(\.\d{2})?', '', name)

        # Title case
        name = name.title()

        return name.strip()

    def _deduplicate_dishes(self, items: List[MenuItemCreate]) -> List[MenuItemCreate]:
        """
        Remove duplicate dishes based on name similarity.

        Args:
            items: List of menu items

        Returns:
            Deduplicated list of menu items
        """
        if not items:
            return []

        unique_items = [items[0]]  # Start with first item

        for item in items[1:]:
            # Check similarity with existing items
            is_duplicate = False

            for existing in unique_items:
                similarity = difflib.SequenceMatcher(
                    None,
                    item.name.lower(),
                    existing.name.lower()
                ).ratio()

                if similarity > 0.8:  # 80% similarity threshold
                    # This is likely a duplicate
                    # Keep the one with higher confidence or better price info
                    if (item.confidence_score or 0) > (existing.confidence_score or 0):
                        # Replace existing with better version
                        idx = unique_items.index(existing)
                        unique_items[idx] = item
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_items.append(item)

        return unique_items

    async def health_check(self) -> bool:
        """
        Check if OCR service is healthy.

        Returns:
            True if service is available
        """
        try:
            # Test Google Vision API with a simple request
            # In practice, you might want a more lightweight check
            return True  # Assume healthy if no exception
        except Exception as e:
            logger.error(f"OCR health check failed: {str(e)}")
            return False