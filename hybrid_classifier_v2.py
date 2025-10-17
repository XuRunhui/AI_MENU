"""Improved hybrid classifier with token budgeting and robust JSON parsing."""

import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

Label = Literal["price", "category", "dish", "description", "junk"]

class LineClassification(BaseModel):
    """Classification result for a line."""
    id: str
    label: Label
    confidence: float = Field(ge=0, le=1)
    method: str

class ImprovedHybridClassifier:
    """Hybrid with token budgeting and robust parsing."""

    # Minimal system prompt (no verbose few-shots)
    ULTRA_SHORT_SYSTEM = """Classify menu OCR lines into: price | category | dish | description | junk
- price: "$12.50", "$5"
- category: generic sections (APPETIZERS, DRINKS, BREAKFAST, DESSERTS)
- dish: specific food items (HUEVOS RANCHEROS, VEGGIE OMELETTE)
ALL CAPS ≠ category! Most dishes are ALL CAPS. Return JSON array: [{"id":"...","label":"...","confidence":0..1}]"""

    # Short system with minimal few-shot
    SHORT_SYSTEM = """Classify menu OCR lines using layout features into: price | category | dish | description | junk

**Layout Features:**
- size_class: "very_large" (2x+ median) | "large" (1.4x+) | "medium_large" (1.1x+) | "medium" (0.9-1.1x) | "small" (0.7-0.9x) | "very_small" (<0.7x)
- relative_height: exact ratio vs median text height
- indent_level: 0=flush left, 1+=indented (in median-height units)
- gap_class: "large" (2x+ gap) | "medium" (0.8-2x) | "small" (<0.8x) | "start" (first in column)
- alignment: "left" or "indented"

**How to Use Layout for Classification:**

1. **Category Detection:**
   - very_large size (2x+ median) → almost always category
   - large size + left alignment + large/start gap → category
   - Example: {"text": "SANDWICHES", "size_class": "very_large", "gap_class": "start"} → category

2. **Dish vs Description (the hard part):**
   - medium/medium_large size + left alignment (indent_level=0) → likely DISH
   - small size + indented (indent_level>0) → likely DESCRIPTION
   - medium size + indented → use content: if ALL CAPS and short → dish, if commas → description
   - Examples:
     * {"text": "FRENCH DIP", "size_class": "medium", "indent_level": 24} → dish (ALL CAPS, short, medium size)
     * {"text": "CUBANO", "size_class": "medium", "indent_level": 24} → dish (single word dish name, medium size)
     * {"text": "pickled onions, aioli", "size_class": "small", "indent_level": 24} → description (small, commas)
     * {"text": "CHICKEN NUGGETS, BACON, GRAVY", "size_class": "small", "indent_level": 0} → could be dish or description - use context

3. **Junk Detection:**
   - Single/double letters (CH, d, Z, O, 1) → junk
   - Promotional phrases (SOLD OUT, Leave us a review) → junk
   - Website URLs, social media → junk

**OCR Correction:**
- "S 14" → likely "$14" (price)
- "1O.99" → likely "10.99" (price, O→0)

**Classification Strategy:**
1. Check size_class first - very_large almost always means category
2. If medium/small, check indent_level - left-aligned more likely dish, indented more likely description
3. Use text content (ALL CAPS, commas, length) to break ties
4. Consider gap_class - large gaps often mark new sections

Examples with reasoning:
{"text": "POUTINES", "size_class": "very_large", "indent_level": 6, "gap_class": "start"}
  → category (very_large size dominates, even though indented)

{"text": "BRAISED BEEF", "size_class": "medium", "indent_level": 0, "gap_class": "small"}
  → dish (medium size, left-aligned, ALL CAPS, 2 words)

{"text": "PORCHETTA", "size_class": "medium", "indent_level": 24, "gap_class": "small"}
  → dish (medium size suggests dish despite indentation, single food name)

{"text": "brown gravy, cheese curds, parsley", "size_class": "small", "indent_level": 24}
  → description (small size + indentation + commas = ingredients)

{"text": "CH", "size_class": "small", "indent_level": 0}
  → junk (2-letter fragment)

Return JSON: [{"id":"...","label":"...","confidence":0..1}]"""

    PRICE_PATTERN = re.compile(r'^\s*(?:US\$|[$£€])?\s*\d{1,3}(?:[.,]\d{2})?\s*$')
    # Some OCR engines misread "$" as "S" or mix zeros with "O"
    MISREAD_PRICE_PREFIX = re.compile(r'^[Ss]\s*\d')

    SECTION_WORDS = {
        "appetizer", "appetizers", "starter", "starters",
        "breakfast", "brunch", "lunch", "dinner",
        "dessert", "desserts", "sweet", "sweets",
        "drink", "drinks", "beverage", "beverages",
        "side", "sides",
        "plate", "plates", "platos",
        "wrap", "wraps", "salad", "salads",
        "special", "specials",
        "omelette", "omelettes",
        "sandwich", "sandwiches",
        "poutine", "poutines",  # Added
        "kids", "menu",  # Added for "Kids Menu"
    }

    HEADER_JUNK_PATTERNS = [
        r'ask\s+about',
        r'\bvegan\s+versions?\b',
        r'^\s*menu\s*$',
        r'\bbreakfast\s+everyday\b',
        r'sold\s+out',
        r'leave\s+us\s+a\s+review',
        r'catering\s+available',
        r'\.com/',
        r'follow\s+us',
        r'^the\s+\w+$',  # "The Kroft" etc
    ]

    # Very short strings are usually OCR errors, but allow 2-char prices like "$5"
    MIN_VALID_LENGTH = 2

    def __init__(self, model_path: str = "/data/models/qwen2.5-14b-instruct-awq"):
        """Initialize classifier."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading LLM on device: {self.device}")

        # Load AWQ quantized model with proper settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

    def aggressive_heuristic(self, text: str, bbox: List[List[float]],
                            column_index: int, ocr_conf: float) -> Optional[Tuple[Label, float, str]]:
        """Aggressive heuristic - only return if CERTAIN."""
        text_clean = re.sub(r'<[^>]+>', '', text).strip()
        text_lower = text_clean.lower()

        if not text_clean:
            return ("junk", 0.99, "empty")

        # 0. JUNK - single character OCR errors
        if len(text_clean) == 1:
            return ("junk", 0.99, "single_char")

        num_words = len(text_clean.split())
        comma_count = text_clean.count(',')

        # 1. JUNK - match obvious patterns first
        junk_patterns = [
            r'catering', r'order\s+online', r'follow\s+us', r'www\.',
            r'https?://', r'@\w+', r'yelp', r'delivery',
            r'copyright|©', r'reserved', r'everyday', r'8:00\s*am'
        ] + self.HEADER_JUNK_PATTERNS
        if any(re.search(p, text_lower) for p in junk_patterns):
            return ("junk", 0.95, "junk_pattern")

        # Normalize common OCR quirks for prices (e.g., "S 14", "1O.99")
        price_candidate = text_clean.replace('O', '0').replace('o', '0')
        if self.MISREAD_PRICE_PREFIX.match(price_candidate):
            price_candidate = "$" + price_candidate[1:]

        # 2. PRICE - very strict pattern only (let LLM handle edge cases)
        if self.PRICE_PATTERN.match(price_candidate):
            return ("price", 0.99, "price_pattern")

        # Only catch obvious multi-digit numbers as prices
        if price_candidate.isdigit() and len(price_candidate) >= 3:
            return ("price", 0.95, "numeric")

        # 3. DESCRIPTION - only very obvious cases (let LLM use layout for rest)
        if num_words >= 10 or comma_count >= 3:
            return ("description", 0.95, "very_long")

        if text_clean and text_clean[0].islower() and num_words >= 6:
            return ("description", 0.90, "lowercase_long")

        # 4. CATEGORY - keep strict section word matching
        tokens = [re.sub(r'[^a-z]', '', tok.lower())
                  for tok in re.split(r'[\s/&-]+', text_clean)
                  if re.sub(r'[^a-z]', '', tok.lower())]
        if tokens and len(tokens) <= 3 and all(tok in self.SECTION_WORDS for tok in tokens):
            return ("category", 0.90, "section_word")

        # 5. DISH - only very obvious ALL CAPS multi-word patterns
        # Let LLM use layout to decide single words and edge cases
        leading_token = re.split(r'[\(-]', text_clean, maxsplit=1)[0].strip()
        has_digits = any(ch.isdigit() for ch in leading_token)

        if leading_token.isupper() and not has_digits:
            word_count = len(leading_token.split())
            # Only catch clear 3-5 word dish names
            if 3 <= word_count <= 5:
                return ("dish", 0.92, "all_caps_multi")

        # AMBIGUOUS - let LLM use layout features to decide
        return None

    def compute_layout_features(self, ocr_results: List[Tuple], column_assignments: List[int]) -> List[Dict]:
        """Extract geometric and layout features from OCR bboxes."""
        features = []

        # Extract raw geometry
        for i, (result, col_idx) in enumerate(zip(ocr_results, column_assignments)):
            bbox, text, conf = result

            # Compute box dimensions
            box_np = np.array(bbox)
            width = np.linalg.norm(box_np[1] - box_np[0])
            height = np.linalg.norm(box_np[3] - box_np[0])
            x_center = box_np[:,0].mean()
            y_center = box_np[:,1].mean()
            x_start = box_np[:,0].min()

            features.append({
                "index": i,
                "column_index": col_idx,
                "width": width,
                "height": height,
                "x_center": x_center,
                "y_center": y_center,
                "x_start": x_start,
                "ocr_conf": conf
            })

        # Normalize heights (relative to median)
        heights = [f["height"] for f in features]
        median_height = np.median(heights)

        for f in features:
            f["relative_height"] = f["height"] / median_height if median_height > 0 else 1.0

            # Multi-level size classification with more granularity
            rh = f["relative_height"]
            if rh > 2.0:
                f["size_class"] = "very_large"  # Category headers, titles
            elif rh > 1.4:
                f["size_class"] = "large"  # Section headers, emphasized text
            elif rh > 1.1:
                f["size_class"] = "medium_large"  # Dish names, important items
            elif rh > 0.9:
                f["size_class"] = "medium"  # Normal dish text
            elif rh > 0.7:
                f["size_class"] = "small"  # Descriptions, details
            else:
                f["size_class"] = "very_small"  # Fine print, notes

        # Compute indentation per column
        for col_idx in set(column_assignments):
            col_features = [f for f in features if f["column_index"] == col_idx]
            if not col_features:
                continue

            # Find leftmost x position in this column
            left_margin = min([f["x_start"] for f in col_features])

            # Compute indent levels
            for f in col_features:
                indent_pixels = f["x_start"] - left_margin
                # Normalize by median height (as unit)
                f["indent_level"] = int(indent_pixels / median_height) if median_height > 0 else 0
                f["alignment"] = "left" if indent_pixels < median_height * 0.3 else "indented"

        # Compute vertical gaps (distance to previous item in same column)
        features.sort(key=lambda f: (f["column_index"], f["y_center"]))

        for i, f in enumerate(features):
            if i > 0 and features[i-1]["column_index"] == f["column_index"]:
                gap = f["y_center"] - features[i-1]["y_center"]
                f["y_gap_prev"] = gap / median_height if median_height > 0 else 0

                # Gap classification
                if f["y_gap_prev"] > 2.0:
                    f["gap_class"] = "large"
                elif f["y_gap_prev"] > 0.8:
                    f["gap_class"] = "medium"
                else:
                    f["gap_class"] = "small"
            else:
                f["y_gap_prev"] = None
                f["gap_class"] = "start"

        # Restore original order
        features.sort(key=lambda f: f["index"])

        return features

    def safe_parse_json_array(self, s: str) -> List[Dict]:
        """Robust JSON extraction."""
        # Strip markdown code fences
        s_clean = re.sub(r'```(?:json)?\s*', '', s).strip()

        # Try full array
        m = re.search(r'\[[\s\S]*\]', s_clean)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                pass

        # Try single object and wrap
        m = re.search(r'\{[\s\S]*\}', s_clean)
        if m:
            try:
                return [json.loads(m.group(0))]
            except:
                pass

        raise ValueError("No valid JSON found")

    def chunk_by_token_budget(self, items: List[Dict], system_msg: str,
                              max_prompt_tokens: int = 2000) -> List[List[Dict]]:
        """Chunk items to stay under token budget."""
        if not items:
            return []

        # Calculate system overhead
        test_prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": ""}],
            tokenize=False, add_generation_prompt=True
        )
        sys_tokens = len(self.tokenizer.encode(test_prompt))

        batches = []
        current_batch = []
        current_tokens = sys_tokens

        for item in items:
            # Estimate tokens for this item
            item_json = json.dumps([item], ensure_ascii=False)
            item_tokens = len(self.tokenizer.encode(item_json))

            if current_batch and current_tokens + item_tokens > max_prompt_tokens:
                # Start new batch
                batches.append(current_batch)
                current_batch = []
                current_tokens = sys_tokens

            current_batch.append(item)
            current_tokens += item_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def build_llm_prompt(self, items: List[Dict], system_msg: str) -> str:
        """Build minimal prompt."""
        user_content = json.dumps(items, ensure_ascii=False)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def llm_classify_batch(self, items: List[Dict], system_msg: str) -> List[Dict]:
        """Classify with LLM, with retry on failure."""
        if not items:
            return []

        prompt = self.build_llm_prompt(items, system_msg)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Deterministic decoding
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        raw = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        print(f"🔍 LLM raw output (first {min(500, len(raw))} chars): {raw[:500]}")

        try:
            parsed = self.safe_parse_json_array(raw)
            if parsed:
                return parsed
        except Exception as e:
            print(f"⚠️  First attempt failed: {e}")
            print(f"Full raw output: {raw}")

        # RETRY with ultra-short system and smaller batches
        print(f"🔄 Retrying with ultra-short prompt and smaller batches...")
        small_batches = self.chunk_by_token_budget(items, self.ULTRA_SHORT_SYSTEM, max_prompt_tokens=800)
        print(f"   Split into {len(small_batches)} smaller batches")
        parsed = []

        for batch_idx, small_batch in enumerate(small_batches):
            print(f"   Retry batch {batch_idx+1}/{len(small_batches)} ({len(small_batch)} items)...")
            prompt2 = self.build_llm_prompt(small_batch, self.ULTRA_SHORT_SYSTEM)
            inputs2 = self.tokenizer(prompt2, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs2 = self.model.generate(
                    **inputs2,
                    max_new_tokens=800,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            raw2 = self.tokenizer.decode(outputs2[0][inputs2['input_ids'].shape[1]:], skip_special_tokens=True)

            print(f"🔍 Retry raw output (first {min(300, len(raw2))} chars): {raw2[:300]}")

            try:
                batch_result = self.safe_parse_json_array(raw2)
                parsed.extend(batch_result)
            except Exception as e2:
                print(f"⚠️  Retry also failed for batch: {e2}")
                print(f"Full retry output: {raw2}")
                # Fallback: guess based on caps
                for item in small_batch:
                    parsed.append({
                        "id": item["id"],
                        "label": "dish" if item.get("is_all_caps") else "junk",
                        "confidence": 0.5
                    })

        return parsed

    def force_price_if_obvious(self, line_text: str, column_role: str,
                              cls: LineClassification) -> LineClassification:
        """Post-LLM fix for obvious prices."""
        text_clean = re.sub(r'<[^>]+>', '', line_text).strip()

        price_candidate = text_clean.replace('O', '0').replace('o', '0')
        if self.MISREAD_PRICE_PREFIX.match(price_candidate):
            price_candidate = "$" + price_candidate[1:]

        if (self.PRICE_PATTERN.match(price_candidate)
                or price_candidate.isdigit()
                or (column_role == "PRICE" and price_candidate.replace('.', '').isdigit())):
            if cls.label != "price":
                return LineClassification(
                    id=cls.id,
                    label="price",
                    confidence=max(cls.confidence, 0.95),
                    method="postfix_price"
                )

        return cls

    def classify_lines(self, ocr_results: List[Tuple], column_assignments: List[int]) -> List[LineClassification]:
        """Main pipeline with token budgeting and layout awareness."""
        classifications = []
        ambiguous = []

        # Compute layout features first
        print("📐 Computing layout features...")
        layout_features = self.compute_layout_features(ocr_results, column_assignments)

        # Phase 1: Aggressive heuristics
        for i, (result, col_idx) in enumerate(zip(ocr_results, column_assignments)):
            bbox, text, conf = result
            line_id = f"line_{i}"

            heuristic_result = self.aggressive_heuristic(text, bbox, col_idx, conf)

            if heuristic_result:
                label, confidence, reason = heuristic_result
                classifications.append(LineClassification(
                    id=line_id,
                    label=label,
                    confidence=confidence,
                    method=f"heuristic_{reason}"
                ))
            else:
                # Ambiguous - enrich with layout features
                text_clean = re.sub(r'<[^>]+>', '', text).strip()
                layout_feat = layout_features[i]

                ambiguous.append({
                    "id": line_id,
                    "text": text_clean,
                    "num_words": len(text_clean.split()),
                    "is_all_caps": text_clean.isupper(),
                    "has_comma": "," in text_clean,
                    "index": i,
                    "column_role": "PRICE" if "$" in text_clean else "NAME",
                    # Add layout features
                    "size_class": layout_feat["size_class"],
                    "relative_height": round(layout_feat["relative_height"], 2),
                    "indent_level": layout_feat["indent_level"],
                    "alignment": layout_feat["alignment"],
                    "gap_class": layout_feat["gap_class"],
                    "y_gap": round(layout_feat["y_gap_prev"], 2) if layout_feat["y_gap_prev"] is not None else None,
                })

        total = len(ocr_results)
        print(f"🎯 Heuristic: {len(classifications)}/{total} ({len(classifications)*100//total}%)")
        print(f"🤖 LLM needed: {len(ambiguous)} ({len(ambiguous)*100//total if total else 0}%)")

        # Phase 2: LLM with token budgeting
        if ambiguous:
            batches = self.chunk_by_token_budget(ambiguous, self.SHORT_SYSTEM, max_prompt_tokens=2000)
            print(f"📦 Split into {len(batches)} token-budgeted batches")

            all_llm_results = []
            for batch_idx, batch in enumerate(batches):
                print(f"  Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} items)...")
                llm_results = self.llm_classify_batch(batch, self.SHORT_SYSTEM)
                all_llm_results.extend(llm_results)

            # Convert to LineClassification with post-fixes
            llm_result_map = {res.get("id"): res for res in all_llm_results if res.get("id")}

            for amb_item in ambiguous:
                llm_result = llm_result_map.get(amb_item["id"])
                if not llm_result:
                    # Fallback: treat all-caps short strings as dishes, otherwise junk
                    fallback_label = "dish" if amb_item["is_all_caps"] else "junk"
                    llm_result = {
                        "id": amb_item["id"],
                        "label": fallback_label,
                        "confidence": 0.5,
                    }

                cls = LineClassification(
                    id=amb_item["id"],
                    label=llm_result["label"],
                    confidence=llm_result.get("confidence", 0.7),
                    method="llm"
                )

                # Apply price post-fix
                orig_text = ocr_results[amb_item["index"]][1]
                cls = self.force_price_if_obvious(orig_text, amb_item["column_role"], cls)

                classifications.append(cls)

        # Sort by original order
        classifications.sort(key=lambda c: int(c.id.split('_')[1]))

        # Print detailed results for each line
        print(f"\n{'='*70}")
        print(f"📋 DETAILED CLASSIFICATION RESULTS")
        print(f"{'='*70}")
        for i, cls in enumerate(classifications):
            # Get original text
            orig_text = ocr_results[i][1]
            text_clean = re.sub(r'<[^>]+>', '', orig_text).strip()

            # Truncate long text for display
            display_text = text_clean[:50] + "..." if len(text_clean) > 50 else text_clean

            print(f"{i:3d}. [{cls.label:11s}] (conf: {cls.confidence:.2f}, method: {cls.method:20s}) | {display_text}")
        print(f"{'='*70}\n")

        # Print summary
        from collections import Counter
        label_counts = Counter(c.label for c in classifications)
        print(f"📊 CLASSIFICATION SUMMARY:")
        for label, count in sorted(label_counts.items()):
            methods = Counter(c.method for c in classifications if c.label == label)
            methods_str = ", ".join(f"{m}:{c}" for m, c in methods.most_common(3))
            print(f"  {label}: {count} ({methods_str})")
        print()

        return classifications
