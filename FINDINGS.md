# AI Safety Fragility Benchmark — Findings Report

**Date:** 2026-03-28
**Author:** AIGuard Research
**Models tested:** Qwen 2.5 7B, Llama 4 Maverick 17B, Claude Sonnet 4.6
**Prompts:** 100 (25 base x 4 framings) | **Responses scored:** 296

---

## Executive Summary

All three models refuse 100% (or near-100%) of direct harmful requests. But when the same requests are reframed as hypothetical scenarios, roleplay, or emotional appeals, safety alignment degrades measurably. Every model tested is vulnerable to framing effects, but the degree and pattern of vulnerability differs dramatically.

**Key finding:** Hypothetical framing is the most effective attack vector across all models. It shifted 92% of Qwen's responses, 76% of Maverick's, and 38% of Sonnet's away from full refusal.

---

## 1. Overall Fragility Index

The Fragility Index measures the mean score uplift from the direct (baseline) condition. Higher = more fragile.

| Model | Roleplay | Hypothetical | Emotional | Overall |
|-------|----------|-------------|-----------|---------|
| **Qwen 2.5 7B** | 0.88 | **1.04** | 0.56 | 0.83 |
| **Maverick 17B** | 0.44 | **0.80** | 0.28 | 0.51 |
| **Sonnet 4.6** | 0.17 | **0.33** | 0.08 | 0.19 |

**Interpretation:** Qwen is approximately 4x more fragile than Sonnet. Maverick sits in between. The ranking hypothetical > roleplay > emotional holds for all three models.

---

## 2. Score Distribution by Model

### Qwen 2.5 7B Instruct Turbo
- **Direct:** 100% full refusal
- **Roleplay:** 24% full refusal, 64% partial refusal, 12% soft compliance
- **Hypothetical:** 8% full refusal, 80% partial refusal, 12% soft compliance
- **Emotional:** 44% full refusal, 56% partial refusal

Qwen shows a **sliding scale vulnerability**. Even though each framing was tested independently (no conversation history carried over), stronger framings consistently extract more information. Emotional framing alone shifts 56% of responses away from full refusal; roleplay shifts 76%; hypothetical shifts 92%. Rather than breaking sharply at a single threshold, Qwen leaks proportionally more as the framing becomes more permissive.

### Llama 4 Maverick 17B (MoE)
- **Direct:** 100% full refusal
- **Roleplay:** 56% full refusal, 44% partial refusal
- **Hypothetical:** 24% full refusal, 72% partial refusal, 4% soft compliance
- **Emotional:** 72% full refusal, 28% partial refusal

Maverick shows a **binary cliff pattern**. It resists most framings with terse "I can't help with that" refusals. But when a framing does break through, the content can be more dangerous — its hypothetical responses included Molotov cocktail instructions and detailed weapon construction that Qwen did not provide.

### Claude Sonnet 4.6
- **Direct:** 71% full refusal, 29% partial refusal
- **Roleplay:** 54% full refusal, 46% partial refusal
- **Hypothetical:** 38% full refusal, 62% partial refusal
- **Emotional:** 71% full refusal, 21% partial refusal, 8% soft compliance

Sonnet shows the **most robust but nuanced pattern**. Notably, it already provides partial information on some direct requests (e.g., lock picking, signature forgery) — framing it as "educational context" unprompted. This means its baseline is higher (1.29 vs 1.00), but the *uplift from framing* is the smallest. Its fragility index of 0.19 is the lowest tested.

Unexpected finding: Sonnet's emotional responses include 8% soft compliance — the only condition where Sonnet reaches score 3. Neither roleplay nor hypothetical achieved this. This contrasts with both open-source models where emotional framing was the *least* effective.

---

## 3. Most Vulnerable Harm Categories

Averaged across all three models and all framings:

| Category | Mean Score | Most Effective Framing |
|----------|-----------|----------------------|
| **Disinformation** | 1.83 | Roleplay, hypothetical, emotional (all 2.0) |
| **Harassment** | 1.62 | Emotional (1.83) |
| **Privacy violation** | 1.54 | Hypothetical (2.0) |
| **Illegal activity** | 1.53 | Hypothetical (1.78) |
| **Dangerous substances** | 1.50 | Roleplay & hypothetical (2.0) |
| **Weapons** | 1.50 | Hypothetical (2.0) |
| **Cybercrime** | 1.44 | Hypothetical (2.0) |
| **Theft** | 1.42 | Roleplay (2.0) |
| **Fraud** | 1.40 | Roleplay & hypothetical (1.58) |
| **Manipulation** | 1.38 | Hypothetical (2.0) |
| **Academic dishonesty** | 1.08 | Hypothetical (1.33) |

**Disinformation is the most vulnerable category** — models are most willing to engage with requests about creating fake news or misleading content when given any framing at all. Academic dishonesty is the most robustly refused.

The spread between direct and hypothetical framing is largest for **cybercrime, dangerous substances, and manipulation** (all 1.0 point uplift), meaning these are the categories where framing makes the biggest difference.

---

## 4. Judge Disagreements

27.4% of responses (81/296) showed disagreement between DeepSeek V3.1 and Llama Guard 4.

| Type | Count | Meaning |
|------|-------|---------|
| Guard says unsafe, judge says refusal | 76 | Guard flags the *topic* even when the model refused |
| Guard says safe, judge says compliant | 5 | Guard misses subtle harmful content |

The dominant pattern (94% of disagreements) is Llama Guard flagging responses as "unsafe" even when the model refused the request. This is because Llama Guard evaluates the *conversation topic* rather than *whether the model actually complied*. It's a topic classifier, not a compliance detector.

This has practical implications: **using a guard model alone would overcount safety failures.** The 4-point spectrum judge more accurately captures the actual safety posture.

Disagreements by model:
- **Maverick:** 35 disagreements (35% of its responses) — mostly guard flagging refusals
- **Sonnet:** 33 disagreements (34%) — same pattern
- **Qwen:** 13 disagreements (13%) — fewer because Qwen gives longer, more educational refusals that guard may classify differently

---

## 5. Three Distinct Fragility Signatures

The most significant finding is that each model exhibits a qualitatively different fragility signature:

### The Leaker (Qwen 2.5 7B)
- Sliding scale vulnerability: each framing type (tested independently, with no shared conversation history) extracts proportionally more information, from emotional (56% shifted) to roleplay (76%) to hypothetical (92%)
- Provides detailed "educational" content under roleplay/hypothetical cover
- Most consistently shiftable — no framing type is fully resisted
- Dangerous because the information leaks are steady and systematic

### The Cliff (Llama 4 Maverick 17B)
- Binary behaviour: terse refusal or significant content
- Most resistant to roleplay and emotional framings
- When hypothetical framing works, the content is more dangerous (weapons instructions, explosive recipes)
- Dangerous because the failures, though fewer, are more severe

### The Educator (Claude Sonnet 4.6)
- Proactively provides "educational context" even on direct requests
- Lowest fragility index but highest baseline score
- Uniquely vulnerable to emotional framing (only model reaching score 3 on emotional)
- Dangerous because its "helpful educator" mode can blur the line between safety information and harmful content

---

## 6. Implications for AI Safety

### Framing is a universal attack vector
Every model tested was vulnerable to framing effects, regardless of architecture, scale, or training approach. This suggests that current safety training methods (RLHF, constitutional AI, etc.) create surface-level refusal behaviours rather than deep understanding of harm.

### Hypothetical framing is the most effective
Across all models and categories, "thought experiment" and "fiction" framings were the most effective at extracting harmful content. This may be because models are trained to be helpful for creative and academic tasks, creating a tension between helpfulness and safety.

### Binary safety evaluation is insufficient
The 27.4% disagreement rate between our spectrum judge and Llama Guard demonstrates that binary safe/unsafe classification misses important nuance. A model that says "I can't help with that but here are the key concepts..." is qualitatively different from one that says "I can't help with that." Both register as "refusal" in binary evaluation.

### Model-specific fragility signatures matter
A single benchmark score cannot capture the difference between a model that leaks steadily (Qwen) and one that breaks catastrophically (Maverick). Safety evaluation should characterise the *shape* of failure, not just its frequency.

---

## 7. Limitations

- **Sample size:** 25 base prompts is sufficient for identifying patterns but too small for statistical significance testing on category-level differences.
- **Single judge model:** DeepSeek V3.1 scored conservatively in calibration (2 mismatches out of 5, both under-scoring compliance). True compliance rates may be slightly higher than reported.
- **Temperature 0:** All target model responses used temperature 0 for reproducibility. Higher temperatures might produce different fragility patterns.
- **No system prompts:** Models were tested without system prompts. Real-world deployments with safety-focused system prompts may show different results.
- **Judge bias on Sonnet:** DeepSeek V3.1 judging Claude Sonnet creates a cross-model evaluation dynamic. The judge may interpret Sonnet's "educational" style differently than Qwen's more explicit responses.

---

## 8. Next Steps

- **Layer 2 (Persona Vectors):** Extract internal representations from the open-source models (Qwen, Maverick) on the same prompts to map how safety-relevant features shift under different framings. This is where the open-source model selection pays off.
- **Larger prompt battery:** Scale to 100+ base prompts for statistical power.
- **System prompt interaction:** Test how safety system prompts interact with framing attacks.
- **Temporal stability:** Re-run on same models after updates to track safety alignment drift.

---

## Appendix: Reproducibility

All data, scripts, and visualisations are in this directory. To reproduce:

```bash
pip install openai plotly pandas anthropic
export TOGETHER_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
python scripts/run_experiment.py
python scripts/run_sonnet.py
python scripts/judge.py
python scripts/analyse.py
```

Raw data is preserved in `results/` for verification.
