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
    SHORT_SYSTEM = """Classify menu OCR into: price | category | dish | description | junk
Rules:
- price: currency/numeric (e.g., "$12", "$5.50")
- category: ONLY generic sections (APPETIZERS, DRINKS, BREAKFAST, OMELETTE, PLATES, SWEETS)
- dish: specific food names (HUEVOS RANCHEROS, CHARLIE'S TACOS, VEGGIE OMELETTE)
- ALL CAPS doesn't mean category! Most dishes are ALL CAPS.

Examples:
"HUEVOS RANCHEROS" → dish
"BREAKFAST" → category
"$12" → price

Return JSON: [{"id":"...","label":"...","confidence":0..1}]"""

    PRICE_PATTERN = re.compile(r'^\s*(?:US\$|\$|£|€)?\s*\d{1,3}(?:[.,]\d{2})?\s*$')

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
        """Aggressive heuristic - only return if CERTAIN."""
        text_clean = re.sub(r'<[^>]+>', '', text).strip()
        text_lower = text_clean.lower()

        if not text_clean:
            return ("junk", 0.99, "empty")

        num_words = len(text_clean.split())
        comma_count = text_clean.count(',')

        # 1. PRICE - very strict
        if self.PRICE_PATTERN.match(text_clean):
            return ("price", 0.99, "price_pattern")

        # 2. DESCRIPTION - long/commas/lowercase
        if num_words >= 8 or comma_count >= 2:
            return ("description", 0.95, "long_or_commas")

        if text_clean and text_clean[0].islower() and num_words >= 4:
            return ("description", 0.90, "lowercase_start")

        # 3. JUNK
        junk_patterns = [
            r'catering', r'order\s+online', r'follow\s+us', r'www\.',
            r'https?://', r'@\w+', r'yelp', r'delivery',
            r'copyright|©', r'reserved', r'everyday', r'8:00\s*am'
        ]
        if any(re.search(p, text_lower) for p in junk_patterns):
            return ("junk", 0.95, "junk_pattern")

        # 4. CATEGORY - only generic section words
        section_words = ['appetizers', 'entrees', 'desserts', 'drinks', 'beverages', 'sides', 'soups', 'salads']
        if (num_words <= 2 and
            any(w in text_lower for w in section_words) and
            text_clean.isupper()):
            return ("category", 0.90, "section_word")

        # AMBIGUOUS
        return None

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

        if self.PRICE_PATTERN.match(text_clean) or (column_role == "PRICE" and text_clean.isdigit()):
            if cls.label != "price":
                return LineClassification(
                    id=cls.id,
                    label="price",
                    confidence=max(cls.confidence, 0.95),
                    method="postfix_price"
                )

        return cls

    def classify_lines(self, ocr_results: List[Tuple], column_assignments: List[int]) -> List[LineClassification]:
        """Main pipeline with token budgeting."""
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
                # Ambiguous
                text_clean = re.sub(r'<[^>]+>', '', text).strip()
                ambiguous.append({
                    "id": line_id,
                    "text": text_clean,
                    "num_words": len(text_clean.split()),
                    "is_all_caps": text_clean.isupper(),
                    "index": i,
                    "column_role": "PRICE" if "$" in text_clean else "NAME"  # Quick guess
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
            for amb_item, llm_result in zip(ambiguous, all_llm_results):
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

        # Print summary
        from collections import Counter
        label_counts = Counter(c.label for c in classifications)
        print(f"\n📊 Final classification:")
        for label, count in sorted(label_counts.items()):
            methods = Counter(c.method for c in classifications if c.label == label)
            methods_str = ", ".join(f"{m}:{c}" for m, c in methods.most_common(3))
            print(f"  {label}: {count} ({methods_str})")

        return classifications
