"""Hybrid classifier: Aggressive heuristics first, LLM only for ambiguous cases."""

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

class HybridMenuClassifier:
    """Aggressive heuristics + LLM for ambiguous cases only."""

    def __init__(self, model_path: str = "/data/models/qwen2.5-14b-instruct-awq"):
        """Initialize classifier."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading LLM on device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

    def aggressive_heuristic(self, text: str, bbox: List[List[float]],
                            column_index: int, ocr_conf: float) -> Optional[Tuple[Label, float, str]]:
        """Aggressive heuristic classification - only return if CERTAIN."""
        # Clean HTML tags
        text_clean = re.sub(r'<[^>]+>', '', text).strip()
        text_lower = text_clean.lower()

        if not text_clean:
            return ("junk", 0.99, "empty")

        num_words = len(text_clean.split())
        comma_count = text_clean.count(',')

        # 1. PRICE - very strict pattern
        if re.match(r'^\s*\$?\s*\d{1,3}([.,]\d{2})?\s*$', text_clean):
            return ("price", 0.99, "price_pattern")

        # 2. DESCRIPTION - long text with commas or starts lowercase
        if num_words >= 8 or comma_count >= 2:
            return ("description", 0.95, "long_or_commas")

        if text_clean and text_clean[0].islower() and num_words >= 4:
            return ("description", 0.90, "lowercase_start")

        # 3. JUNK - promotional text
        junk_patterns = [
            r'catering', r'order\s+online', r'follow\s+us', r'www\.',
            r'https?://', r'@\w+', r'yelp', r'delivery',
            r'copyright|©', r'reserved', r'everyday', r'8:00\s*am'
        ]
        if any(re.search(p, text_lower) for p in junk_patterns):
            return ("junk", 0.95, "junk_pattern")

        # 4. CATEGORY - only if generic section word + certain characteristics
        section_words = ['appetizers', 'entrees', 'desserts', 'drinks',
                        'beverages', 'sides', 'soups', 'salads']
        if (num_words <= 2 and
            any(w in text_lower for w in section_words) and
            text_clean.isupper()):
            return ("category", 0.90, "section_word")

        # AMBIGUOUS - needs LLM
        return None

    def build_llm_prompt(self, ambiguous_lines: List[Dict[str, Any]]) -> str:
        """Build prompt for ambiguous cases only."""
        system_prompt = """You classify OCR text from restaurant menus into:
- price: currency amounts (e.g., "$12.50")
- category: generic section headers ONLY (e.g., "APPETIZERS", "ENTREES", "DRINKS", "BREAKFAST", "OMELETTE", "PLATES", "WRAPS/SALADS", "SWEETS")
- dish: menu item names (e.g., "HUEVOS RANCHEROS", "CHARLIE'S TACOS", "VEGGIE OMELETTE")
- description: already filtered out
- junk: already filtered out

CRITICAL RULES:
- ALL CAPS does NOT mean category! Most dishes are in ALL CAPS.
- "HUEVOS RANCHEROS" = dish (specific food), "BREAKFAST" = category (generic section)
- "OMELETTE" alone = category, "VEGGIE OMELETTE" = dish
- "PLATES" alone = category, "CHARLIE'S TACOS" = dish
- If it sounds like a specific food item, it's a dish

Few-shot:
Input: {"text":"HUEVOS RANCHEROS","num_words":2,"is_all_caps":true}
Output: {"label":"dish","confidence":0.95}

Input: {"text":"BREAKFAST","num_words":1,"is_all_caps":true}
Output: {"label":"category","confidence":0.90}

Input: {"text":"$12","num_words":1,"is_all_caps":false}
Output: {"label":"price","confidence":0.95}

Input: {"text":"VEGGIE (served with potatoes)","num_words":4,"is_all_caps":false}
Output: {"label":"dish","confidence":0.92}

Return ONLY JSON array: [{"id":"...","label":"...","confidence":0..1}]"""

        user_prompt = f"""Classify these ambiguous lines:
{json.dumps(ambiguous_lines, ensure_ascii=False)}

Return ONLY JSON array."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def llm_classify_batch(self, ambiguous_lines: List[Dict]) -> List[Dict]:
        """Classify ambiguous lines with LLM."""
        if not ambiguous_lines:
            return []

        prompt = self.build_llm_prompt(ambiguous_lines)
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

        # Extract JSON
        start = raw.find('[')
        end = raw.rfind(']')
        payload = raw[start:end+1] if start != -1 and end != -1 else '[]'
        print(f"The prompt was: {prompt}")
        print(payload)
        try:
            return json.loads(payload)
        except json.JSONDecodeError as e:
            print(f"⚠️  LLM parse error: {e}")
            print(f"Response: {raw[:200]}")
            # Fallback: guess dish for ALL CAPS, junk for others
            return [
                {"id": line["id"],
                 "label": "dish" if line.get("is_all_caps") else "junk",
                 "confidence": 0.5}
                for line in ambiguous_lines
            ]

    def classify_lines(self, ocr_results: List[Tuple], column_assignments: List[int]) -> List[LineClassification]:
        """Main classification pipeline."""
        classifications = []
        ambiguous = []

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
                # Ambiguous - needs LLM
                text_clean = re.sub(r'<[^>]+>', '', text).strip()
                ambiguous.append({
                    "id": line_id,
                    "text": text_clean,
                    "num_words": len(text_clean.split()),
                    "is_all_caps": text_clean.isupper(),
                    "index": i
                })

        print(f"🎯 Heuristic classified: {len(classifications)}/{len(ocr_results)} lines ({len(classifications)*100//len(ocr_results)}%)")
        print(f"🤖 LLM needed for: {len(ambiguous)} ambiguous lines ({len(ambiguous)*100//len(ocr_results)}%)")

        # Phase 2: LLM for ambiguous cases
        if ambiguous:
            llm_results = self.llm_classify_batch(ambiguous)

            for amb_line, llm_result in zip(ambiguous, llm_results):
                classifications.append(LineClassification(
                    id=amb_line["id"],
                    label=llm_result["label"],
                    confidence=llm_result.get("confidence", 0.7),
                    method="llm"
                ))

        # Sort by original order
        classifications.sort(key=lambda c: int(c.id.split('_')[1]))

        # Print classification summary
        from collections import Counter
        label_counts = Counter(c.label for c in classifications)
        print(f"\n📊 Final classification:")
        for label, count in sorted(label_counts.items()):
            method_counts = Counter(c.method for c in classifications if c.label == label)
            methods_str = ", ".join(f"{m}:{c}" for m, c in method_counts.items())
            print(f"  {label}: {count} ({methods_str})")

        return classifications
