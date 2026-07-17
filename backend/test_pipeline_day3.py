#!/usr/bin/env python3
"""
test_pipeline_day3.py — Day 3 state machine and AI validation gate CLI verification.

Runs the complete Kairos pipeline end-to-end including Gate 1 and Gate 2 AI checks,
retry loops, and best unedited frame fallbacks.

Saves the decoded result to `/test-data/composite_day3.jpg`.

Usage:
    py -X utf8 test_pipeline_day3.py <path_to_burst_directory>
"""

import argparse
import base64
import logging
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.pipeline.state_machine import PipelineCoordinator, jobs_db
from app import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_pipeline_day3")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Day 3 Pipeline Coordinator and AI Validation Gates."
    )
    parser.add_argument(
        "burst_dir",
        type=str,
        help="Path to the directory containing the burst photos"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Disable demo-fallback mode and hit the live APIs (requires GEMINI_API_KEY / ANTHROPIC_API_KEY)"
    )
    args = parser.parse_args()

    burst_dir = os.path.abspath(args.burst_dir)
    
    # Toggle live vs fallback mode
    if args.live:
        config.DEMO_FALLBACK = False
        logger.info("Running in LIVE API mode. Checking API keys...")
        if config.GATE_PROVIDER == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            logger.error("GEMINI_API_KEY not found in environment. Exiting.")
            sys.exit(1)
        elif config.GATE_PROVIDER == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            logger.error("ANTHROPIC_API_KEY not found in environment. Exiting.")
            sys.exit(1)
    else:
        config.DEMO_FALLBACK = True
        logger.info("Running in OFFLINE DEMO-FALLBACK mode. Will use cached response templates.")

    # Create a unique burst ID
    burst_id = "test-burst-day3"
    
    # Instantiate the coordinator
    coordinator = PipelineCoordinator()
    
    logger.info("Starting pipeline coordinator for burst ID: %s", burst_id)
    start_time = time.time()
    
    # Run the pipeline synchronously for test logging clarity
    coordinator.run_pipeline(burst_id, burst_dir)
    
    elapsed = time.time() - start_time
    logger.info("Pipeline coordinator finished in %.2f seconds.", elapsed)
    
    # Verify the results recorded in jobs_db
    if burst_id not in jobs_db:
        logger.error("Job record not found in jobs_db. Pipeline coordinator failed to run.")
        sys.exit(1)
        
    job = jobs_db[burst_id]
    result = job.get("result")
    
    print("\n" + "=" * 80)
    print(" KAIROS DAY 3 STATE MACHINE RESULTS ".center(80, "="))
    print("=" * 80)
    print(f"Status               : {job['status']}")
    print(f"Final Message        : {job['message']}")
    print(f"Progress Percentage  : {job['progress_percentage']}%")
    
    if result:
        print(f"Result Type          : {result.result_type}")
        print(f"Retry Count          : {result.retry_count}")
        
        # Gate 1 Details
        if result.gate_1_result:
            g1 = result.gate_1_result
            print(f"Gate 1 (Inputs) Pass : {g1.passed} (Confidence: {g1.confidence:.2f})")
            if g1.issues:
                for issue in g1.issues:
                    print(f"  - Issue: [{issue.issue_type}] {issue.description}")
        else:
            print("Gate 1 (Inputs) Pass : N/A (Bypassed)")

        # Gate 2 Details
        if result.gate_2_result:
            g2 = result.gate_2_result
            print(f"Gate 2 (Output) Pass : {g2.passed} (Confidence: {g2.confidence:.2f})")
            if g2.issues:
                for issue in g2.issues:
                    person = issue.person_cluster_id or "global"
                    print(f"  - Issue on {person}: [{issue.issue_type}] {issue.description}")
        else:
            print("Gate 2 (Output) Pass : N/A (Bypassed)")

        # Save result image
        if result.output_image_url:
            # Decode the base64 output image url
            header, encoded = result.output_image_url.split(",", 1)
            img_data = base64.b64decode(encoded)
            
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            out_path = os.path.join(repo_root, "test-data", "composite_day3.jpg")
            
            with open(out_path, "wb") as f_out:
                f_out.write(img_data)
            
            print(f"\n✓ Decoded composite image written to: {out_path}")
            
    print("=" * 80)


if __name__ == "__main__":
    main()
