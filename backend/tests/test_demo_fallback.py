"""
test_demo_fallback.py — Verify DEMO_FALLBACK environment variable behaviour.

When DEMO_FALLBACK=true the AIGateways class must:
  - Return a passing GateResult for Gate 1 by reading app/cache/gate1_response.json.
  - Return a passing GateResult for Gate 2 by reading app/cache/gate2_response.json.
  - Never make any external HTTP/API calls (neither Gemini nor Anthropic clients
    should be instantiated or invoked).
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_cache_dir(gate1: dict, gate2: dict) -> str:
    """Write two cache JSON files to a temporary directory and return its path."""
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "gate1_response.json"), "w") as f:
        json.dump(gate1, f)
    with open(os.path.join(d, "gate2_response.json"), "w") as f:
        json.dump(gate2, f)
    return d


def _make_gateways(cache_dir: str):
    """Instantiate AIGateways with DEMO_FALLBACK forced on and a custom cache dir."""
    # Patch config so DEMO_FALLBACK is True inside the module
    with patch("app.pipeline.gates.config") as mock_cfg:
        mock_cfg.DEMO_FALLBACK = True
        mock_cfg.GATE_PROVIDER = "gemini"
        mock_cfg.GEMINI_MODEL = "gemini-1.5-flash"
        mock_cfg.ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"

        from app.pipeline.gates import AIGateways
        gw = AIGateways()
        # Override the cache directory to our temp dir so we control the JSON
        gw._cache_dir = cache_dir
        gw._demo_fallback = True
        return gw


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_gate1_uses_cache_when_demo_fallback_true():
    """Gate 1 should read the cache file and never call any LLM API."""
    cache_dir = _make_cache_dir(
        gate1={"valid": True, "person_count_estimate": 4, "issues": []},
        gate2={"natural": True, "confidence": 0.95, "flagged_people": []},
    )

    with patch("app.pipeline.gates.config") as mock_cfg:
        mock_cfg.DEMO_FALLBACK = True
        mock_cfg.GATE_PROVIDER = "gemini"
        mock_cfg.GEMINI_MODEL = "gemini-1.5-flash"
        mock_cfg.ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"

        from app.pipeline.gates import AIGateways
        gw = AIGateways()
        gw._cache_dir = cache_dir
        gw._demo_fallback = True

        dummy_images = [np.zeros((10, 10, 3), dtype=np.uint8)]

        # Patch both provider call methods to assert they are never invoked
        with patch.object(gw, "_call_gemini", side_effect=AssertionError("LLM called in demo mode")) as mock_gemini, \
             patch.object(gw, "_call_anthropic", side_effect=AssertionError("LLM called in demo mode")) as mock_anthropic:

            result = gw.check_gate1_inputs(dummy_images)

    assert result.passed is True
    assert result.confidence == 1.0
    assert result.issues == []


def test_gate2_uses_cache_when_demo_fallback_true():
    """Gate 2 should read the cache file and never call any LLM API."""
    cache_dir = _make_cache_dir(
        gate1={"valid": True, "person_count_estimate": 4, "issues": []},
        gate2={"natural": True, "confidence": 0.93, "flagged_people": []},
    )

    with patch("app.pipeline.gates.config") as mock_cfg:
        mock_cfg.DEMO_FALLBACK = True
        mock_cfg.GATE_PROVIDER = "gemini"
        mock_cfg.GEMINI_MODEL = "gemini-1.5-flash"
        mock_cfg.ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"

        from app.pipeline.gates import AIGateways
        gw = AIGateways()
        gw._cache_dir = cache_dir
        gw._demo_fallback = True

        composite = np.zeros((50, 50, 3), dtype=np.uint8)
        base = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(gw, "_call_gemini", side_effect=AssertionError("LLM called in demo mode")), \
             patch.object(gw, "_call_anthropic", side_effect=AssertionError("LLM called in demo mode")):

            result = gw.check_gate2_output(composite, base)

    assert result.passed is True
    assert result.confidence == pytest.approx(0.93, abs=1e-3)
    assert result.issues == []


def test_gate1_cache_with_issues_fails():
    """If the gate1 cache reports issues, the GateResult.passed should be False."""
    cache_dir = _make_cache_dir(
        gate1={
            "valid": False,
            "person_count_estimate": 1,
            "issues": [{"type": "insufficient_faces", "description": "Only 1 face found."}],
        },
        gate2={"natural": True, "confidence": 0.95, "flagged_people": []},
    )

    with patch("app.pipeline.gates.config") as mock_cfg:
        mock_cfg.DEMO_FALLBACK = True
        mock_cfg.GATE_PROVIDER = "gemini"
        mock_cfg.GEMINI_MODEL = "gemini-1.5-flash"
        mock_cfg.ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"

        from app.pipeline.gates import AIGateways
        gw = AIGateways()
        gw._cache_dir = cache_dir
        gw._demo_fallback = True

        result = gw.check_gate1_inputs([np.zeros((10, 10, 3), dtype=np.uint8)])

    assert result.passed is False
    assert len(result.issues) == 1
    assert result.issues[0].issue_type == "insufficient_faces"


def test_gate2_cache_with_flagged_people_fails():
    """If the gate2 cache flags people, GateResult.passed should be False."""
    cache_dir = _make_cache_dir(
        gate1={"valid": True, "person_count_estimate": 4, "issues": []},
        gate2={
            "natural": False,
            "confidence": 0.35,
            "flagged_people": [
                {
                    "person_cluster_id": "person_2",
                    "issue_type": "seam_artifact",
                    "description": "Visible seam along left cheek.",
                }
            ],
        },
    )

    with patch("app.pipeline.gates.config") as mock_cfg:
        mock_cfg.DEMO_FALLBACK = True
        mock_cfg.GATE_PROVIDER = "gemini"
        mock_cfg.GEMINI_MODEL = "gemini-1.5-flash"
        mock_cfg.ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"

        from app.pipeline.gates import AIGateways
        gw = AIGateways()
        gw._cache_dir = cache_dir
        gw._demo_fallback = True

        composite = np.zeros((50, 50, 3), dtype=np.uint8)
        base = np.zeros((50, 50, 3), dtype=np.uint8)
        result = gw.check_gate2_output(composite, base)

    assert result.passed is False
    assert result.confidence == pytest.approx(0.35, abs=1e-3)
    assert len(result.issues) == 1
    assert result.issues[0].person_cluster_id == "person_2"
    assert result.issues[0].issue_type == "seam_artifact"
