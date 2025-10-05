"""LLM-based menu line classifier using heuristics + small LLM for ambiguous cases."""

import re
import json
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, AwqConfig
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

class LineClassification(BaseModel):
    """Classification result for a line."""
    id: str
    label: Label
    confidence: float = Field(ge=0, le=1)
    method: str  # "heuristic" or "llm"

class MenuLineClassifier:
    """Hybrid classifier using heuristics + LLM for ambiguous cases."""

    # Junk patterns
    JUNK_PATTERNS = [
        r'catering\s+available',
        r'order\s+online',
        r'follow\s+us',
        r'www\.',
        r'https?://',
        r'@\w+',  # social handles
        r'yelp',
        r'delivery\s+available',
        r'copyright|©',
        r'all\s+rights\s+reserved',
    ]

    def __init__(self, model_path: str = "/data/models/qwen2.5-14b-instruct-awq"):
        """Initialize the classifier with the LLM model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Determine device
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

        # Check if mostly digits
        digit_count = sum(1 for c in text_clean if c.isdigit())
        is_mostly_digits = digit_count > (total_chars * 0.5)

        comma_count = text_clean.count(',')

        # Calculate bbox center and height
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

    def heuristic_classify(self, feat: LineFeatures) -> Optional[LineClassification]:
        """Apply heuristic rules. Returns None if ambiguous (needs LLM)."""
        text_lower = feat.text.lower()

        # Rule 1: Price detection
        if feat.has_currency or feat.is_mostly_digits:
            # Check if it looks like a price
            price_pattern = r'^\$?\s*\d{1,3}(?:[.,]\d{2})?$'
            if re.match(price_pattern, feat.text.strip()):
                return LineClassification(
                    id=feat.id,
                    label="price",
                    confidence=0.95,
                    method="heuristic"
                )

        # Rule 2: Junk detection
        if self.junk_re.search(text_lower):
            return LineClassification(
                id=feat.id,
                label="junk",
                confidence=0.90,
                method="heuristic"
            )

        # Rule 3: Description detection
        if feat.num_words >= 8 or feat.comma_count >= 2:
            return LineClassification(
                id=feat.id,
                label="description",
                confidence=0.85,
                method="heuristic"
            )

        # Rule 4: Category/section header detection - VERY conservative
        # Only auto-classify as category if it's a well-known section word
        known_sections = ['menu', 'appetizers', 'entrees', 'desserts', 'drinks', 'beverages', 'sides']
        text_lower = feat.text.lower().strip()

        if (feat.is_all_caps and
            feat.num_words <= 3 and
            not feat.has_currency and
            any(section in text_lower for section in known_sections)):
            return LineClassification(
                id=feat.id,
                label="category",
                confidence=0.80,
                method="heuristic"
            )

        # Ambiguous - needs LLM
        return None

    def build_llm_prompt(self, lines: List[Dict[str, Any]]) -> str:
        """Build the LLM classification prompt."""
        system_prompt = """You classify OCR text lines from restaurant menus into one of:
- price: currency amounts or numeric price tokens (e.g., "$12.50", "13.95").
- category: section headers ONLY for generic categories (e.g., "APPETIZERS", "ENTREES", "DRINKS", "DESSERTS"). Must be generic menu sections, NOT specific dish names.
- dish: the menu item name - this is the actual food/drink being sold (e.g., "HUEVOS RANCHEROS", "CHARLIE'S TACOS", "VEGGIE OMELETTE"). Can be ALL CAPS or Title Case. Usually 1-7 words, mostly letters.
- description: details or ingredients for the dish (often longer, has commas, describes ingredients).
- junk: promotional or unrelated text (e.g., "Catering available", "Order online", URLs, social media).

IMPORTANT Rules:
- Dish names can be ALL CAPS! Don't confuse them with categories.
- Category = generic section (APPETIZERS, DRINKS). Dish = specific food item (HUEVOS RANCHEROS, PANCAKES).
- If text looks like a food name, even if ALL CAPS, label it "dish".
- Prefer "price" when numeric/currency dominates.
- If very long or many commas, prefer "description".
- Use "junk" for social/contact/promo/legal strings.

Return strict JSON array. For each item: {"id": str, "label": "<price|category|dish|description|junk>", "confidence": 0..1}"""

        user_prompt = f"""Classify these lines. Each has features and limited context.

EXAMPLES of correct classification:
- "HUEVOS RANCHEROS" → dish (it's a specific food item, even though ALL CAPS)
- "DRINKS" → category (generic section, not a specific item)
- "$15" or "$ 12" → price (standalone price)
- "eggs, corn, beans, salsa" → description (ingredients list)

{json.dumps(lines, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON array with classifications. Be strict: most ALL CAPS items are dishes, not categories."""

        # Format for Qwen
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def llm_classify_batch(self, features: List[LineFeatures],
                          context_map: Dict[str, Dict[str, Optional[str]]]) -> List[LineClassification]:
        """Classify ambiguous lines using LLM."""
        # Build input for LLM
        llm_input = []
        for feat in features:
            context = context_map.get(feat.id, {})
            llm_input.append({
                "id": feat.id,
                "text": feat.text,
                "features": {
                    "num_words": feat.num_words,
                    "is_all_caps": feat.is_all_caps,
                    "has_currency": feat.has_currency,
                    "comma_count": feat.comma_count,
                    "alpha_ratio": round(feat.alpha_ratio, 2)
                },
                "context": context
            })

        # Build prompt
        prompt = self.build_llm_prompt(llm_input)

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Parse JSON response
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                classifications = json.loads(json_match.group())
                return [
                    LineClassification(
                        id=c["id"],
                        label=c["label"],
                        confidence=c.get("confidence", 0.7),
                        method="llm"
                    )
                    for c in classifications
                ]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  LLM JSON parse error: {e}")
            print(f"Response: {response[:200]}")

        # Fallback: mark as junk with low confidence
        return [
            LineClassification(id=f.id, label="junk", confidence=0.3, method="llm_fallback")
            for f in features
        ]

    def build_context_map(self, features: List[LineFeatures]) -> Dict[str, Dict[str, Optional[str]]]:
        """Build context (prev/next in same column) for each line."""
        # Group by column
        by_column = {}
        for feat in features:
            if feat.column_index not in by_column:
                by_column[feat.column_index] = []
            by_column[feat.column_index].append(feat)

        # Sort by y_center within each column
        for col_feats in by_column.values():
            col_feats.sort(key=lambda f: f.y_center)

        # Build context map
        context_map = {}
        for col_idx, col_feats in by_column.items():
            for i, feat in enumerate(col_feats):
                prev_text = col_feats[i-1].text if i > 0 else None
                next_text = col_feats[i+1].text if i < len(col_feats) - 1 else None
                context_map[feat.id] = {
                    "prev_in_column": prev_text,
                    "next_in_column": next_text
                }

        return context_map

    def classify_lines(self, ocr_results: List[Dict[str, Any]],
                      column_assignments: List[int]) -> List[LineClassification]:
        """Classify all OCR lines using hybrid approach."""
        # Extract features
        features = []
        for i, (result, col_idx) in enumerate(zip(ocr_results, column_assignments)):
            bbox, text, conf = result
            feat = self.extract_features(
                line_id=f"line_{i}",
                text=text,
                bbox=bbox,
                column_index=col_idx,
                ocr_confidence=conf
            )
            features.append(feat)

        # Build context
        context_map = self.build_context_map(features)

        # Classify using heuristics
        classifications = []
        ambiguous_features = []

        for feat in features:
            heuristic_result = self.heuristic_classify(feat)
            if heuristic_result:
                classifications.append(heuristic_result)
            else:
                ambiguous_features.append(feat)

        print(f"🎯 Heuristic classified: {len(classifications)}/{len(features)} lines")
        print(f"🤖 LLM needed for: {len(ambiguous_features)} ambiguous lines")

        # Classify ambiguous using LLM (in batches)
        if ambiguous_features:
            batch_size = 50
            for i in range(0, len(ambiguous_features), batch_size):
                batch = ambiguous_features[i:i+batch_size]
                llm_results = self.llm_classify_batch(batch, context_map)
                classifications.extend(llm_results)

        # Sort by original order
        classifications.sort(key=lambda c: int(c.id.split('_')[1]))

        return classifications
