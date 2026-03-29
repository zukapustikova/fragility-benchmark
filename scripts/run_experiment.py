"""Phase 3a: Run 100 prompts × 2 target models = 200 API calls. Save raw responses."""

import json
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TOGETHER_API_KEY,
    TOGETHER_BASE_URL,
    TARGET_MODELS,
    PROMPTS_DIR,
    RESULTS_DIR,
)
from openai import OpenAI

client = OpenAI(base_url=TOGETHER_BASE_URL, api_key=TOGETHER_API_KEY)


def query_model(model_id, prompt_text, max_retries=3):
    """Send a single prompt to a model and return the response."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=512,
                temperature=0.0,  # deterministic for reproducibility
            )
            return {
                "response": resp.choices[0].message.content,
                "finish_reason": resp.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                },
            }
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for {model_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
    return {"response": None, "finish_reason": "error", "usage": {}}


def main():
    # Load battery
    battery_path = os.path.join(PROMPTS_DIR, "battery.json")
    with open(battery_path) as f:
        battery = json.load(f)

    print(f"Loaded {len(battery)} prompts")
    print(f"Target models: {TARGET_MODELS}")
    print(f"Total API calls: {len(battery) * len(TARGET_MODELS)}")
    print()

    # Check for existing partial results (resume support)
    results_path = os.path.join(RESULTS_DIR, "raw_responses.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        done_keys = {(r["prompt_id"], r["model"]) for r in results}
        print(f"Resuming: {len(done_keys)} responses already collected")
    else:
        results = []
        done_keys = set()

    total = len(battery) * len(TARGET_MODELS)
    completed = len(done_keys)

    for model_id in TARGET_MODELS:
        model_short = model_id.split("/")[-1]
        print(f"\n--- {model_short} ---")

        for i, prompt in enumerate(battery):
            key = (prompt["id"], model_id)
            if key in done_keys:
                continue

            completed += 1
            print(f"  [{completed}/{total}] {prompt['id']} ({prompt['condition']})", end="", flush=True)

            result = query_model(model_id, prompt["text"])

            results.append({
                "prompt_id": prompt["id"],
                "base_id": prompt["base_id"],
                "condition": prompt["condition"],
                "condition_id": prompt["condition_id"],
                "category": prompt["category"],
                "prompt_text": prompt["text"],
                "model": model_id,
                "model_short": model_short,
                "response": result["response"],
                "finish_reason": result["finish_reason"],
                "usage": result["usage"],
                "timestamp": datetime.utcnow().isoformat(),
            })

            # Save after every 10 responses (crash recovery)
            if completed % 10 == 0:
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)

            print(f" ✓" if result["response"] else " ✗")
            time.sleep(0.3)  # rate limiting

    # Final save
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    errors = sum(1 for r in results if r["response"] is None)
    print(f"\n=== COMPLETE ===")
    print(f"Total responses: {len(results)}")
    print(f"Errors: {errors}")
    print(f"Saved to: {results_path}")


if __name__ == "__main__":
    main()
