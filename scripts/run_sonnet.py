"""Run the 100-prompt battery through Claude Sonnet via Anthropic API."""

import json
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROMPTS_DIR, RESULTS_DIR

import anthropic

client = anthropic.Anthropic()
MODEL_ID = "claude-sonnet-4-6"
MODEL_SHORT = "Claude-Sonnet-4.6"


def query_model(prompt_text, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=MODEL_ID,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return {
                "response": resp.content[0].text,
                "finish_reason": resp.stop_reason,
                "usage": {
                    "prompt_tokens": resp.usage.input_tokens,
                    "completion_tokens": resp.usage.output_tokens,
                },
            }
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return {"response": None, "finish_reason": "error", "usage": {}}


def main():
    # Load battery
    with open(os.path.join(PROMPTS_DIR, "battery.json")) as f:
        battery = json.load(f)

    # Load existing results to append Sonnet
    results_path = os.path.join(RESULTS_DIR, "raw_responses.json")
    with open(results_path) as f:
        results = json.load(f)

    # Check what's already done for Sonnet
    done_keys = {(r["prompt_id"], r["model"]) for r in results if r["model"] == MODEL_ID}
    print(f"Loaded {len(battery)} prompts, {len(done_keys)} already done for {MODEL_SHORT}")

    total = len(battery)
    completed = len(done_keys)

    print(f"\n--- {MODEL_SHORT} ---")
    for i, prompt in enumerate(battery):
        key = (prompt["id"], MODEL_ID)
        if key in done_keys:
            continue

        completed += 1
        print(f"  [{completed}/{total}] {prompt['id']} ({prompt['condition']})", end="", flush=True)

        result = query_model(prompt["text"])

        results.append({
            "prompt_id": prompt["id"],
            "base_id": prompt["base_id"],
            "condition": prompt["condition"],
            "condition_id": prompt["condition_id"],
            "category": prompt["category"],
            "prompt_text": prompt["text"],
            "model": MODEL_ID,
            "model_short": MODEL_SHORT,
            "response": result["response"],
            "finish_reason": result["finish_reason"],
            "usage": result["usage"],
            "timestamp": datetime.utcnow().isoformat(),
        })

        if completed % 10 == 0:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        print(f" ✓" if result["response"] else " ✗")
        time.sleep(0.3)

    # Final save
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    errors = sum(1 for r in results if r["model"] == MODEL_ID and r["response"] is None)
    sonnet_count = sum(1 for r in results if r["model"] == MODEL_ID)
    print(f"\n=== COMPLETE ===")
    print(f"Sonnet responses: {sonnet_count}")
    print(f"Errors: {errors}")
    print(f"Total responses in file: {len(results)}")


if __name__ == "__main__":
    main()
