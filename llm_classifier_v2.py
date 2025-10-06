"""Improved LLM-based menu line classifier with geometry signals and deterministic decoding."""

import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

Label = Literal["price", "category", "dish", "description", "junk"]

class LineFeatures(BaseModel):
    """Features extracted from an OCR line."""
    id: str
    text: str
    bbox: List[List[float]]
    column_index: int
    y_center: float
    num_words: int
    alpha_ratio: float
    is_all_caps: bool
    has_currency: bool
    is_mostly_digits: bool
    comma_count: int
    bbox_height: float
    ocr_confidence: float

class GeometrySignals(BaseModel):
    """Geometry signals for LLM classification."""
    column_role: str  # PRICE / NAME / OTHER
    price_in_same_row: bool
    dist_to_nearest_price_y: float
    dist_to_nearest_price_x: float
    nearest_above_category: Optional[str]
    height_rel: float
    leaders_in_row: bool

class LineClassification(BaseModel):
    """Classification result for a line."""
    id: str
    label: Label
    confidence: float = Field(ge=0, le=1)
    method: str  # "heuristic", "llm", or "postfix"

class ImprovedMenuLineClassifier:
    """Improved classifier with geometry signals and deterministic decoding."""

    JUNK_PATTERNS = [
        r'catering\s+available', r'order\s+online', r'follow\s+us', r'www\.',
        r'https?://', r'@\w+', r'yelp', r'delivery\s+available',
        r'copyright|©', r'all\s+rights\s+reserved',
    ]

    SECTION_LEXICON = {
        'appetizers', 'soups', 'salads', 'entrees', 'mains', 'sides',
        'desserts', 'drinks', 'beers', 'wines', 'breakfast', 'lunch',
        'dinner', 'specials', 'house specialties', 'omelette', 'plates',
        'wraps', 'sweets', 'beverages'
    }

    def __init__(self, model_path: str = "/data/models/qwen2.5-14b-instruct-awq"):
        """Initialize the classifier."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading LLM on device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
        self.junk_re = re.compile('|'.join(self.JUNK_PATTERNS), re.IGNORECASE)

    def extract_features(self, line_id: str, text: str, bbox: List[List[float]],
                        column_index: int, ocr_confidence: float) -> LineFeatures:
        """Extract features from an OCR line."""
        # Clean HTML tags first
        text_clean = re.sub(r'<[^>]+>', '', text).strip()

        # Calculate features
        num_words = len(text_clean.split())
        alpha_chars = sum(1 for c in text_clean if c.isalpha())
        total_chars = len(text_clean) or 1
        alpha_ratio = alpha_chars / total_chars

        is_all_caps = text_clean.isupper() if text_clean else False
        has_currency = '$' in text_clean or '€' in text_clean or '£' in text_clean

        digit_count = sum(1 for c in text_clean if c.isdigit())
        is_mostly_digits = digit_count > (total_chars * 0.5)

        comma_count = text_clean.count(',')

        y_center = (bbox[0][1] + bbox[2][1]) / 2
        bbox_height = bbox[2][1] - bbox[0][1]

        return LineFeatures(
            id=line_id,
            text=text_clean,
            bbox=bbox,
            column_index=column_index,
            y_center=y_center,
            num_words=num_words,
            alpha_ratio=alpha_ratio,
            is_all_caps=is_all_caps,
            has_currency=has_currency,
            is_mostly_digits=is_mostly_digits,
            comma_count=comma_count,
            bbox_height=bbox_height,
            ocr_confidence=ocr_confidence
        )

    def compute_geometry_signals(self, features: List[LineFeatures]) -> Dict[str, GeometrySignals]:
        """Compute geometry signals for each line."""
        # Calculate median line height
        heights = [f.bbox_height for f in features]
        median_height = np.median(heights) if heights else 1.0

        # Identify price lines
        price_lines = []
        for f in features:
            if re.match(r'^\s*\$?\s*\d{1,3}([.,]\d{2})?\s*$', f.text):
                price_lines.append(f)

        # Detect column roles based on content
        column_roles = {}
        for col_idx in set(f.column_index for f in features):
            col_features = [f for f in features if f.column_index == col_idx]
            price_count = sum(1 for f in col_features if re.match(r'^\s*\$?\s*\d{1,3}([.,]\d{2})?\s*$', f.text))

            if price_count > len(col_features) * 0.5:
                column_roles[col_idx] = "PRICE"
            else:
                # Check if mostly text (dish names)
                text_count = sum(1 for f in col_features if f.alpha_ratio > 0.7)
                column_roles[col_idx] = "NAME" if text_count > len(col_features) * 0.5 else "OTHER"

        # Find nearest category above each line
        category_candidates = [f for f in features if f.is_all_caps and f.num_words <= 3]
        category_candidates.sort(key=lambda x: x.y_center)

        # Compute signals for each feature
        signals = {}
        for f in features:
            # Find nearest price
            min_dist_y = float('inf')
            min_dist_x = float('inf')
            price_in_row = False

            for p in price_lines:
                dist_y = abs(p.y_center - f.y_center)
                dist_x = abs(p.bbox[0][0] - f.bbox[0][0])

                if dist_y < min_dist_y:
                    min_dist_y = dist_y
                if dist_x < min_dist_x:
                    min_dist_x = dist_x

                if dist_y < 10:  # Same row threshold
                    price_in_row = True

            # Find nearest category above
            nearest_cat = None
            for cat in category_candidates:
                if cat.y_center < f.y_center:
                    nearest_cat = cat.text
                else:
                    break

            # Check for leaders (dots)
            leaders_in_row = bool(re.search(r'\.{2,}', f.text))

            signals[f.id] = GeometrySignals(
                column_role=column_roles.get(f.column_index, "OTHER"),
                price_in_same_row=price_in_row,
                dist_to_nearest_price_y=min_dist_y if min_dist_y != float('inf') else 999,
                dist_to_nearest_price_x=min_dist_x if min_dist_x != float('inf') else 999,
                nearest_above_category=nearest_cat,
                height_rel=round(f.bbox_height / median_height, 2),
                leaders_in_row=leaders_in_row
            )

        return signals

    def heuristic_classify(self, feat: LineFeatures, geom: GeometrySignals) -> Optional[LineClassification]:
        """Apply heuristic rules with geometry signals."""
        text_lower = feat.text.lower()

        # Rule 1: Price (improved with geometry)
        if geom.column_role == "PRICE" and (feat.has_currency or feat.is_mostly_digits):
            if re.match(r'^\s*\$?\s*\d{1,3}([.,]\d{2})?\s*$', feat.text):
                return LineClassification(id=feat.id, label="price", confidence=0.98, method="heuristic")

        # Rule 2: Junk
        if self.junk_re.search(text_lower):
            return LineClassification(id=feat.id, label="junk", confidence=0.95, method="heuristic")

        # Rule 3: Description (long, has commas, or follows dish closely)
        if feat.num_words >= 8 or feat.comma_count >= 2:
            return LineClassification(id=feat.id, label="description", confidence=0.90, method="heuristic")

        # Rule 4: Category (only if in lexicon and far from prices)
        if (any(kw in text_lower for kw in self.SECTION_LEXICON) and
            feat.is_all_caps and feat.num_words <= 3 and
            geom.dist_to_nearest_price_y > 50 and geom.height_rel > 1.2):
            return LineClassification(id=feat.id, label="category", confidence=0.85, method="heuristic")

        # Ambiguous - needs LLM
        return None

    def build_llm_prompt(self, lines_with_geometry: List[Dict[str, Any]]) -> str:
        """Build improved prompt with geometry signals and few-shot examples."""
        system_prompt = """You label OCR lines from restaurant menus into one of:
- price | category | dish | description | junk

Signals you receive (per line):
- text, features (num_words, is_all_caps, has_currency, comma_count, alpha_ratio, height_rel),
- geometry: column_index, column_role (PRICE|NAME|OTHER), price_in_same_row (bool),
  dist_to_nearest_price_y, dist_to_nearest_price_x, nearest_above_category (string or null).

Rules:
- "category" = generic section headers only (APPETIZERS, ENTREES, DESSERTS, DRINKS, SIDES, BREAKFAST, LUNCH, DINNER, HOUSE SPECIALTIES).
- ALL CAPS can be a dish; do NOT mark a specific item as "category".
- Prefer "price" if text is numeric/currency or column_role==PRICE and text is short numeric.
- Prefer "dish" if in NAME column near a price (small dist_to_nearest_price_y).
- Prefer "description" if long or comma-heavy and follows a dish in same column.
- "junk" for social/promo/URL/legal.

Return ONLY strict JSON array: [{"id": "...", "label": "...", "confidence": 0..1}]

Few-shot examples:
Input: {"id":"ex1","text":"HUEVOS RANCHEROS","features":{"num_words":2,"is_all_caps":true,"comma_count":0,"alpha_ratio":1.0,"height_rel":1.2},"geometry":{"column_role":"NAME","price_in_same_row":false,"dist_to_nearest_price_y":6,"dist_to_nearest_price_x":180,"nearest_above_category":"BREAKFAST"}}
Output: [{"id":"ex1","label":"dish","confidence":0.95}]

Input: {"id":"ex2","text":"$12.50","features":{"num_words":1,"is_all_caps":false,"comma_count":0,"alpha_ratio":0.0,"height_rel":0.8},"geometry":{"column_role":"PRICE","price_in_same_row":true,"dist_to_nearest_price_y":0,"dist_to_nearest_price_x":0,"nearest_above_category":null}}
Output: [{"id":"ex2","label":"price","confidence":0.99}]

Input: {"id":"ex3","text":"HOUSE SPECIALTIES","features":{"num_words":2,"is_all_caps":true,"comma_count":0,"alpha_ratio":1.0,"height_rel":1.6},"geometry":{"column_role":"OTHER","price_in_same_row":false,"dist_to_nearest_price_y":120,"dist_to_nearest_price_x":300,"nearest_above_category":null}}
Output: [{"id":"ex3","label":"category","confidence":0.93}]

Input: {"id":"ex4","text":"Catering available for parties","features":{"num_words":4,"is_all_caps":false,"comma_count":0,"alpha_ratio":0.95,"height_rel":1.0},"geometry":{"column_role":"OTHER","price_in_same_row":false,"dist_to_nearest_price_y":600,"dist_to_nearest_price_x":600,"nearest_above_category":null}}
Output: [{"id":"ex4","label":"junk","confidence":0.98}]"""

        user_prompt = f"""Classify these lines with their geometry signals:

{json.dumps(lines_with_geometry, ensure_ascii=False)}

Return ONLY a JSON array."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def llm_classify_batch(self, features_with_geom: List[tuple]) -> List[LineClassification]:
        """Classify with LLM using deterministic decoding."""
        # Build input with geometry
        llm_input = []
        for feat, geom in features_with_geom:
            llm_input.append({
                "id": feat.id,
                "text": feat.text,
                "features": {
                    "num_words": feat.num_words,
                    "is_all_caps": feat.is_all_caps,
                    "has_currency": feat.has_currency,
                    "comma_count": feat.comma_count,
                    "alpha_ratio": round(feat.alpha_ratio, 2),
                    "height_rel": geom.height_rel
                },
                "geometry": {
                    "column_index": feat.column_index,
                    "column_role": geom.column_role,
                    "price_in_same_row": geom.price_in_same_row,
                    "dist_to_nearest_price_y": int(geom.dist_to_nearest_price_y),
                    "dist_to_nearest_price_x": int(geom.dist_to_nearest_price_x),
                    "nearest_above_category": geom.nearest_above_category
                }
            })

        prompt = self.build_llm_prompt(llm_input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Deterministic decoding
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        raw = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Safer JSON extraction
        start = raw.find('[')
        end = raw.rfind(']')
        payload = raw[start:end+1] if start != -1 and end != -1 else '[]'
        print(f"The prompt was: {prompt}")
        print(payload)
        try:
            classifications = json.loads(payload)
            return [
                LineClassification(id=c["id"], label=c["label"], confidence=c.get("confidence", 0.7), method="llm")
                for c in classifications
            ]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  LLM JSON parse error: {e}")
            print(f"Raw response: {raw[:300]}")
            return [LineClassification(id=f.id, label="junk", confidence=0.3, method="llm_fallback")
                   for f, _ in features_with_geom]

    def post_llm_sanity(self, feat: LineFeatures, geom: GeometrySignals,
                       cls: LineClassification, median_height: float) -> LineClassification:
        """Apply post-LLM sanity checks."""
        text_lower = feat.text.strip().lower()

        # Fix 1: Category not in lexicon + near price = likely dish
        if cls.label == "category" and not any(s in text_lower for s in self.SECTION_LEXICON):
            if geom.dist_to_nearest_price_y < 1.5 * median_height:
                return LineClassification(id=feat.id, label="dish", confidence=0.75, method="postfix")

        # Fix 2: Dish but very long/comma-heavy = likely description
        if cls.label == "dish" and (feat.num_words >= 10 or feat.comma_count >= 2):
            return LineClassification(id=feat.id, label="description", confidence=0.75, method="postfix")

        # Fix 3: Price but doesn't match pattern = junk
        if cls.label == "price" and not re.match(r'^\s*\$?\s*\d{1,3}([.,]\d{2})?\s*$', feat.text.strip()):
            if geom.column_role != "PRICE":
                return LineClassification(id=feat.id, label="junk", confidence=0.65, method="postfix")

        return cls

    def classify_lines(self, ocr_results: List[tuple], column_assignments: List[int]) -> List[LineClassification]:
        """Main classification pipeline."""
        # Extract features
        features = []
        for i, (result, col_idx) in enumerate(zip(ocr_results, column_assignments)):
            bbox, text, conf = result
            feat = self.extract_features(f"line_{i}", text, bbox, col_idx, conf)
            features.append(feat)

        # Compute geometry signals
        print("🔍 Computing geometry signals...")
        geometry_signals = self.compute_geometry_signals(features)
        median_height = np.median([f.bbox_height for f in features])

        # Heuristic pass
        classifications = []
        ambiguous = []

        for feat in features:
            geom = geometry_signals[feat.id]
            heuristic_result = self.heuristic_classify(feat, geom)
            if heuristic_result:
                classifications.append(heuristic_result)
            else:
                ambiguous.append((feat, geom))

        print(f"🎯 Heuristic classified: {len(classifications)}/{len(features)} lines")
        print(f"🤖 LLM needed for: {len(ambiguous)} ambiguous lines")

        # LLM pass
        if ambiguous:
            batch_size = 50
            for i in range(0, len(ambiguous), batch_size):
                batch = ambiguous[i:i+batch_size]
                llm_results = self.llm_classify_batch(batch)

                # Post-LLM sanity
                for (feat, geom), cls in zip(batch, llm_results):
                    cls = self.post_llm_sanity(feat, geom, cls, median_height)
                    classifications.append(cls)

        # Sort by original order
        classifications.sort(key=lambda c: int(c.id.split('_')[1]))

        return classifications
