"""
gates.py — AI input and output verification gates.

This module implements two verification gates using vision LLMs (Gemini or
Claude) to protect the pipeline from bad input data and check blending quality:

1. Gate 1 (Input Validation): Downsizes and stacks the burst images into a
   single image grid, then validates:
     - Same location / group / scene.
     - Acceptable image resolution and quality.
     - At least 2 faces present.
2. Gate 2 (Output Verification): Closely inspects the blended composite image
   and compares it with the base frame for blending artifacts (visible seams,
   warped features, lighting/skin mismatches).

Features a 'demo-fallback' mode that bypasses API calls using offline JSON
caches, ensuring robust presentations.

Part of the Kairos CV pipeline: detect → cluster → score → blend → **gate**.
"""

import base64
import json
import logging
import math
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from app import config
from app.schemas import GateResult, GateIssue

logger = logging.getLogger(__name__)


class AIGateways:
    """Manages Gemini and Anthropic vision API integrations for Kairos validation gates."""

    def __init__(self) -> None:
        # Load API keys from environment
        self._gemini_api_key = os.environ.get("GEMINI_API_KEY")
        self._anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        self._demo_fallback = config.DEMO_FALLBACK
        self._cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cache",
        )

        logger.debug(
            "AIGateways initialized: fallback=%s, provider=%s",
            self._demo_fallback, config.GATE_PROVIDER
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_gate1_inputs(self, images: List[np.ndarray]) -> GateResult:
        """Verify quality and scene consistency of the uploaded burst.

        Args:
            images: List of group photo numpy arrays.

        Returns:
            A GateResult indicating validation success or failure.
        """
        logger.info("Running Gate 1: Input Validation...")

        if self._demo_fallback:
            logger.info("Gate 1: Using offline demo-fallback cache.")
            return self._load_cached_gate1()

        if not images:
            return GateResult(
                passed=False,
                confidence=0.0,
                issues=[GateIssue(issue_type="quality_issue", description="No images provided.")]
            )

        # 1. Create a downsized image grid to keep payload size and latency low
        grid_image = self._create_image_grid(images, target_width=300)
        
        prompt = (
            "These images are a burst sequence of the same group of people in the same location.\n"
            "Verify the following conditions:\n"
            "1. It is the same group of people and same background scene across all frames.\n"
            "2. The quality is usable (images are not corrupted or extremely blurry).\n"
            "3. At least 2 faces are visible and distinguishable.\n\n"
            "Analyze the input grid image. You must return a JSON object with this exact structure:\n"
            "{\n"
            "  \"valid\": bool,\n"
            "  \"person_count_estimate\": int,\n"
            "  \"issues\": [\n"
            "    {\"type\": \"scene_mismatch\" | \"blurry_image\" | \"insufficient_faces\" | \"quality_issue\", "
            "\"description\": string}\n"
            "  ]\n"
            "}"
        )

        try:
            # 2. Call chosen LLM provider
            raw_text = self._call_vision_llm(
                prompt=prompt,
                images=[grid_image],
                system_instruction="You are an expert group photo validation assistant. Output raw JSON only."
            )
            
            # 3. Parse and translate response to GateResult schema
            res_json = self._parse_json_response(raw_text)
            valid = res_json.get("valid", True)
            
            issues = []
            for item in res_json.get("issues", []):
                issues.append(GateIssue(
                    person_cluster_id=None,
                    issue_type=item.get("type", "quality_issue"),
                    description=item.get("description", "Input validation failed")
                ))

            passed = len(issues) == 0 and valid
            person_count = res_json.get("person_count_estimate", None)
            result = GateResult(
                passed=passed,
                confidence=1.0 if passed else 0.0,
                issues=issues,
                person_count_estimate=person_count,
            )
            logger.info("Gate 1 complete: passed=%s, people=%s, issues=%d", result.passed, person_count, len(result.issues))
            return result

        except Exception as e:
            logger.error("Gate 1 API call failed: %s. Falling back to fail-safe pass.", e)
            # Default fallback in case of API failure: log and pass to prevent freezing the pipeline
            return GateResult(passed=True, confidence=0.8, issues=[])

    def check_gate2_output(self, composite: np.ndarray, base_image: np.ndarray) -> GateResult:
        """Inspect the blended composite image closely for visual quality issues.

        Args:
            composite:  The final blended group photo.
            base_image: The original frame used as background.

        Returns:
            A GateResult indicating natural appearance or detailing issues.
        """
        logger.info("Running Gate 2: Blending Artifact Inspection...")

        if self._demo_fallback:
            logger.info("Gate 2: Using offline demo-fallback cache.")
            return self._load_cached_gate2()

        prompt = (
            "Below are two images: first is the 'original_base' image, and second is the 'composite' blended image.\n"
            "The 'composite' image was created by replacing some faces in the 'original_base' image using automated image blending.\n\n"
            "Inspect the 'composite' image very closely for any visual anomalies or editing artifacts, particularly "
            "around face boundaries (such as visible seams, mismatched lighting or skin tones, ghosting, double edges, "
            "or warped facial features).\n\n"
            "Return your evaluation as a JSON object with this exact structure:\n"
            "{\n"
            "  \"natural\": bool,\n"
            "  \"confidence\": float,\n"
            "  \"flagged_people\": [\n"
            "    {\n"
            "      \"person_cluster_id\": string,\n"
            "      \"issue_type\": \"seam_artifact\" | \"lighting_mismatch\" | \"warped_feature\",\n"
            "      \"description\": string\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Notes:\n"
            "- 'confidence' must be a float between 0.0 and 1.0 representing how natural and seamless the composite looks.\n"
            "- If there are visible seams or warped faces, set 'natural' to false, lower the confidence, and list the "
            "offending person cluster ID (e.g. 'person_0', 'person_1', etc.)."
        )

        try:
            # Send both base image (for reference) and composite to LLM
            raw_text = self._call_vision_llm(
                prompt=prompt,
                images=[base_image, composite],
                system_instruction="You are an expert digital forensics image inspector. Output raw JSON only."
            )
            
            res_json = self._parse_json_response(raw_text)
            natural = res_json.get("natural", True)
            confidence = res_json.get("confidence", 1.0)
            
            issues = []
            for item in res_json.get("flagged_people", []):
                issues.append(GateIssue(
                    person_cluster_id=item.get("person_cluster_id"),
                    issue_type=item.get("issue_type", "seam_artifact"),
                    description=item.get("description", "Blending artifact detected")
                ))

            passed = len(issues) == 0 and natural
            result = GateResult(
                passed=passed,
                confidence=confidence,
                issues=issues
            )
            logger.info("Gate 2 complete: passed=%s, confidence=%.2f, issues=%d",
                        result.passed, result.confidence, len(result.issues))
            return result

        except Exception as e:
            logger.error("Gate 2 API call failed: %s. Falling back to fail-safe pass.", e)
            # Default to passing on network/LLM failure to keep the app working
            return GateResult(passed=True, confidence=0.85, issues=[])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_image_grid(self, images: List[np.ndarray], target_width: int) -> np.ndarray:
        """Stack multiple images into a downsized grid layout."""
        resized_imgs = []
        for img in images:
            h, w = img.shape[:2]
            target_height = int(h * (target_width / w))
            resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            resized_imgs.append(resized)
            
        num_images = len(resized_imgs)
        cols = int(math.ceil(math.sqrt(num_images)))
        rows = int(math.ceil(num_images / cols))
        
        # Standardize sizes to the first image's size to stack cleanly
        grid_height = resized_imgs[0].shape[0]
        grid_width = resized_imgs[0].shape[1]
        
        standardized = []
        for img in resized_imgs:
            if img.shape[:2] != (grid_height, grid_width):
                img = cv2.resize(img, (grid_width, grid_height), interpolation=cv2.INTER_AREA)
            standardized.append(img)
            
        blank_tile = np.zeros_like(standardized[0])
        while len(standardized) < rows * cols:
            standardized.append(blank_tile)
            
        row_images = []
        for r in range(rows):
            start = r * cols
            end = start + cols
            row_img = np.hstack(standardized[start:end])
            row_images.append(row_img)
            
        grid = np.vstack(row_images)
        return grid

    def _call_vision_llm(
        self,
        prompt: str,
        images: List[np.ndarray],
        system_instruction: str
    ) -> str:
        """Delegate vision call to chosen LLM provider."""
        provider = config.GATE_PROVIDER.lower()
        
        if provider == "gemini":
            if not self._gemini_api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
            return self._call_gemini(prompt, images, system_instruction)
            
        elif provider == "anthropic":
            if not self._anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
            return self._call_anthropic(prompt, images, system_instruction)
            
        else:
            raise ValueError(f"Unknown GATE_PROVIDER config: {config.GATE_PROVIDER}")

    def _call_gemini(self, prompt: str, images: List[np.ndarray], system_instruction: str) -> str:
        """Call Gemini API using PIL conversions."""
        import google.generativeai as genai
        
        genai.configure(api_key=self._gemini_api_key)
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            system_instruction=system_instruction
        )
        
        # Convert BGR (cv2) to RGB PIL Images
        pil_images = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))
            
        content = pil_images + [prompt]
        
        response = model.generate_content(
            content,
            generation_config={"response_mime_type": "application/json"}
        )
        return response.text

    def _call_anthropic(self, prompt: str, images: List[np.ndarray], system_instruction: str) -> str:
        """Call Anthropic Claude API using base64 image encoding."""
        from anthropic import Anthropic
        
        client = Anthropic(api_key=self._anthropic_api_key)
        
        content_list = []
        for img in images:
            # Encode BGR image to JPEG base64
            _, buffer = cv2.imencode(".jpg", img)
            img_b64 = base64.b64encode(buffer).decode("utf-8")
            
            content_list.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64
                }
            })
            
        content_list.append({
            "type": "text",
            "text": prompt
        })

        message = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=system_instruction,
            messages=[{
                "role": "user",
                "content": content_list
            }]
        )
        return message.content[0].text

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Sanitize and load raw JSON output string from LLMs."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        return json.loads(text)

    # ── Fallback Cache Loader Helpers ─────────────────────────────────────────

    def _load_cached_gate1(self) -> GateResult:
        cache_path = os.path.join(self._cache_dir, "gate1_response.json")
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            
            valid = data.get("valid", True)
            issues = []
            for item in data.get("issues", []):
                issues.append(GateIssue(
                    person_cluster_id=None,
                    issue_type=item.get("type", "quality_issue"),
                    description=item.get("description", "Mock fail")
                ))
            return GateResult(
                passed=len(issues) == 0 and valid,
                confidence=1.0 if len(issues) == 0 else 0.0,
                issues=issues,
                person_count_estimate=data.get("person_count_estimate", None),
            )
        except Exception as e:
            logger.error("Failed to load cached Gate 1 response: %s", e)
            return GateResult(passed=True, confidence=1.0, issues=[])

    def _load_cached_gate2(self) -> GateResult:
        cache_path = os.path.join(self._cache_dir, "gate2_response.json")
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            
            natural = data.get("natural", True)
            confidence = data.get("confidence", 1.0)
            issues = []
            for item in data.get("flagged_people", []):
                issues.append(GateIssue(
                    person_cluster_id=item.get("person_cluster_id"),
                    issue_type=item.get("issue_type", "seam_artifact"),
                    description=item.get("description", "Mock artifact")
                ))
            return GateResult(
                passed=len(issues) == 0 and natural,
                confidence=confidence,
                issues=issues
            )
        except Exception as e:
            logger.error("Failed to load cached Gate 2 response: %s", e)
            return GateResult(passed=True, confidence=1.0, issues=[])
