"""
SAE Token-Level Trace: Measure feature activations at EVERY output token.

Purpose: Determine whether violation-predictive features are active
DURING compliance language generation, not just at the prompt level.

This answers: does the model "know" it will violate while writing
"meets all criteria," or does the violation decision emerge later?

Methodology:
1. Feed healthcare prompts known to produce both violations and clean decisions
2. Generate the response ONE TOKEN AT A TIME
3. After each token, capture SAE feature activations at layers 31 and 53
4. Tag each reasoning token as compliance-asserting or not
5. Compare violation-predictive feature activations across:
   (a) Compliance-asserting tokens vs other reasoning tokens in violations
   (b) Decision token vs reasoning tokens in violations
   (c) Violation vs clean cases at matched compliance-asserting positions
6. Optional: intervention timing test (steer during reasoning vs at decision)

Output: Per-prompt JSONL with full token trajectories for downstream analysis.

Requires: 1x A100 80GB, Gemma 3 27B-IT, Gemma Scope 2 SAEs.
Runtime: ~45-90 minutes for 50 cases.
"""
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
import json
import time
import re
import numpy as np


# ─────────────────────────────────────────────────────────
# KNOWN VIOLATION-PREDICTIVE FEATURES (from gemma3_sae_probe.py)
# ─────────────────────────────────────────────────────────

TRACKED_FEATURES = {
    # From gemma3_sae_probe.py results (sae_results_gemma3_27b.json)
    # SAE: gemma-scope-2-27b-it-res, layer_XX_width_16k_l0_small
    # Positive d = elevated in violations, negative d = elevated in clean
    31: {
        6095: "violation_elevated_d2.13",   # viol=1295 clean=273
        1382: "violation_elevated_d2.12",   # viol=926 clean=196
        28:   "violation_elevated_d2.12",   # viol=533 clean=112
        3205: "violation_elevated_d2.11",   # viol=705 clean=149
        7648: "violation_elevated_d2.06",   # viol=573 clean=125
        5785: "violation_elevated_d2.04",   # viol=512 clean=113
        3269: "violation_elevated_d2.03",   # viol=430 clean=95
        6649: "violation_elevated_d1.94",   # viol=1125 clean=294
        373:  "violation_elevated_d1.92",   # viol=1341 clean=322
        3501: "violation_elevated_d1.90",   # viol=447 clean=106
        184:  "violation_elevated_d1.68",   # viol=2909 clean=901
        635:  "clean_elevated_d1.25",       # viol=33 clean=452 (SUPPRESSED in violations)
    },
    53: {
        170:   "violation_elevated_d2.37",  # viol=2606 clean=478
        14516: "violation_elevated_d2.20",  # viol=3049 clean=617
        9305:  "violation_elevated_d2.18",  # viol=2736 clean=561
        13310: "violation_elevated_d2.06",  # viol=2512 clean=547
        4745:  "violation_elevated_d2.05",  # viol=1438 clean=315
        293:   "violation_elevated_d2.01",  # viol=6247 clean=1397
        3541:  "violation_elevated_d1.91",  # viol=5468 clean=1476
        838:   "clean_elevated_d1.20",      # viol=172 clean=1552 (SUPPRESSED)
        870:   "clean_elevated_d1.17",      # viol=336 clean=2865 (SUPPRESSED)
        2236:  "clean_elevated_d0.92",      # viol=0 clean=686 (SUPPRESSED)
        1687:  "violation_elevated_d1.59",  # viol=900 clean=166
        11368: "violation_elevated_d1.18",  # viol=886 clean=133
    },
}

TOP_K_FEATURES = 20  # Also track top-K active features per token


# ─────────────────────────────────────────────────────────
# COMPLIANCE LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────

COMPLIANCE_KEYWORDS = [
    "met", "meets", "confirmed", "consistent", "appropriate", "compliant",
    "criteria", "criterion", "section 110", "section110", "regulatory",
    "stable", "qualified", "documented", "established", "satisf",
    "support", "medically necessary", "medical necessity",
    "mbpm", "cms", "medicare", "coverage", "authorized", "approve",
    "demonstrates", "tolerance", "supervision", "interdisciplinary",
    "rehabilitation potential", "functional improvement",
]

def is_compliance_asserting(token_text, cumulative_text):
    """Check if a token is part of compliance-asserting language.

    Checks the token itself AND a small window of surrounding context
    from the cumulative text to catch multi-token compliance phrases.
    """
    t_lower = token_text.lower().strip()
    # Direct token match
    for kw in COMPLIANCE_KEYWORDS:
        if kw in t_lower:
            return True

    # Context window: check last 50 chars of cumulative text
    ctx = cumulative_text[-50:].lower() if len(cumulative_text) > 50 else cumulative_text.lower()
    for kw in COMPLIANCE_KEYWORDS:
        if kw in ctx and t_lower and len(t_lower) > 1:
            # Token is near compliance language
            if any(c.isalpha() for c in t_lower):
                return True

    return False


# ─────────────────────────────────────────────────────────
# REGION CLASSIFICATION
# ─────────────────────────────────────────────────────────

def classify_token_region(cumulative_text):
    """Classify which region of the JSON output we're in."""
    ct = cumulative_text.lower()

    # Check for decision field
    decision_idx = ct.rfind('"decision"')
    reasoning_idx = ct.rfind('"reasoning"')

    if decision_idx > 0 and decision_idx > reasoning_idx:
        return "decision"
    elif reasoning_idx > 0:
        # Check if we've moved past reasoning to another field
        after_reasoning = ct[reasoning_idx:]
        post_fields = ['"confidence"', '"clinical_basis"', '"denial_reasons"',
                       '"los_recommendation"', '"approved_los_days"',
                       '"alternative_level_of_care"', '"conditions"']
        for field in post_fields:
            if field in after_reasoning:
                # Check if decision comes after this field
                if decision_idx > 0:
                    return "decision"
                return "post_reasoning"
        return "reasoning"
    else:
        return "pre_reasoning"


def parse_decision(text):
    """Extract decision from generated JSON text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        d = json.loads(text)
        return d.get("decision", "unknown"), d
    except:
        m = re.search(r'"decision"\s*:\s*"([^"]+)"', text)
        if m:
            return m.group(1), {}
        return "parse_error", {}


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE

    print("=" * 60)
    print("SAE TOKEN-LEVEL TRACE v2")
    print("Per-token compliance tagging + trajectory output")
    print("=" * 60)
    sys.stdout.flush()

    MODEL_ID = "google/gemma-3-27b-it"

    # ── Load model ──
    print("\n[1] Loading Gemma 3 27B-IT...")
    sys.stdout.flush()
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        text_backbone = model.model.language_model
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        text_backbone = model.model
    else:
        raise RuntimeError(f"Cannot find transformer layers in {type(model)}")

    total_layers = len(text_backbone.layers)
    device = next(model.parameters()).device
    print(f"  Loaded in {time.time()-t0:.0f}s, {total_layers} layers, device={device}")
    sys.stdout.flush()

    # ── Load SAEs ──
    print("\n[2] Loading Gemma Scope 2 SAEs...")
    sys.stdout.flush()
    SAE_RELEASE = "gemma-scope-2-27b-it-res"
    target_layers = [31, 53]

    saes = {}
    for layer_idx in target_layers:
        try:
            sae = SAE.from_pretrained(
                release=SAE_RELEASE,
                sae_id=f"layer_{layer_idx}_width_16k_l0_small",
            )
            sae.eval()
            sae.to(device)
            saes[layer_idx] = sae
            print(f"  Layer {layer_idx}: {sae.cfg.d_sae} features")
        except Exception as e:
            print(f"  Layer {layer_idx} FAILED: {e}")
    sys.stdout.flush()

    if not saes:
        print("FATAL: No SAEs loaded.")
        return

    # ── Build prompts ──
    print("\n[3] Building prompts...")
    sys.stdout.flush()
    sys.path.insert(0, "/workspace")
    import healthcare

    system_prompt = healthcare.build_system_prompt()
    nf, g = healthcare.make_prng(42)

    prompts = []
    for i in range(100):
        pair = healthcare.generate_pair(nf, g)
        up = healthcare.build_user_prompt(
            "HRW", "positive_reward", pair, 21, "twin", False
        )
        prompts.append({
            "idx": len(prompts),
            "condition": "HRW",
            "tier": pair["cert_tier"],
            "prompt": up,
        })
        if len(prompts) >= 50:
            break

    print(f"  Built {len(prompts)} HRW twin prompts")
    sys.stdout.flush()

    # ── Token-by-token generation with SAE tracing ──
    print(f"\n[4] Running token-level traces on {len(prompts)} prompts...")
    sys.stdout.flush()

    outpath = "/workspace/mechanistic/results/sae_token_trace_v2.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    outfile = open(outpath, "w", encoding="utf-8")

    MAX_TOKENS = 400
    n_violations = 0
    n_clean = 0

    for prompt_idx, p in enumerate(prompts):
        t_start = time.time()
        print(f"\n  --- Prompt {prompt_idx+1}/{len(prompts)} "
              f"(tier={p['tier']}) ---", end="", flush=True)

        messages = [
            {"role": "user", "content": system_prompt + "\n\n" + p["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        )["input_ids"].to(device)

        # Storage for per-token data
        tokens_data = []
        generated_text = ""
        current_ids = input_ids.clone()

        for step in range(MAX_TOKENS):
            # Hook to capture hidden states
            captured = {}

            def make_hook(layer_idx):
                def hook_fn(module, inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[layer_idx] = h[0, -1, :].detach()
                return hook_fn

            hooks = []
            for layer_idx in saes:
                hook = text_backbone.layers[layer_idx].register_forward_hook(
                    make_hook(layer_idx)
                )
                hooks.append(hook)

            with torch.no_grad():
                outputs = model(current_ids)
                next_token_logits = outputs.logits[0, -1, :]

            for h in hooks:
                h.remove()

            # Sample next token
            probs = torch.softmax(next_token_logits / 0.3, dim=-1)
            next_token = torch.multinomial(probs, 1)
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
            generated_text += token_text

            if next_token.item() == tokenizer.eos_token_id:
                break

            # SAE feature activations
            feature_activations = {}
            for layer_idx, sae in saes.items():
                if layer_idx not in captured:
                    continue
                h = captured[layer_idx].unsqueeze(0)
                with torch.no_grad():
                    feat_acts = sae.encode(h)[0].cpu().float().numpy()

                layer_key = f"L{layer_idx}"

                # Tracked features
                tracked = {}
                if layer_idx in TRACKED_FEATURES:
                    for feat_id, feat_name in TRACKED_FEATURES[layer_idx].items():
                        if feat_id < len(feat_acts):
                            tracked[str(feat_id)] = round(float(feat_acts[feat_id]), 2)

                # Top-K active
                top_idx = np.argsort(feat_acts)[-TOP_K_FEATURES:][::-1]
                top_k = {}
                for i in top_idx:
                    if feat_acts[i] > 0:
                        top_k[str(int(i))] = round(float(feat_acts[i]), 2)

                feature_activations[layer_key] = {
                    "tracked": tracked,
                    "top_k": top_k,
                }

            # Classify region and compliance
            region = classify_token_region(generated_text)
            compliance = is_compliance_asserting(token_text, generated_text) if region == "reasoning" else False

            tokens_data.append({
                "pos": step,
                "token": token_text,
                "region": region,
                "compliance_asserting": compliance,
                "features": feature_activations,
            })

            current_ids = torch.cat(
                [current_ids, next_token.unsqueeze(0)], dim=1
            )

            if step % 50 == 0 and step > 0:
                torch.cuda.empty_cache()

        # Parse decision
        decision, parsed = parse_decision(generated_text)
        is_violation = decision in ("deny", "authorize_with_modification")
        if is_violation:
            n_violations += 1
        else:
            n_clean += 1

        # ── Compute per-region summaries ──
        region_summaries = {}
        for entry in tokens_data:
            region = entry["region"]
            comp = entry["compliance_asserting"]
            # Sub-key: region or region+compliance
            keys_to_update = [region]
            if region == "reasoning":
                keys_to_update.append(f"reasoning_{'compliance' if comp else 'non_compliance'}")

            for rkey in keys_to_update:
                if rkey not in region_summaries:
                    region_summaries[rkey] = {"count": 0, "features": {}}
                region_summaries[rkey]["count"] += 1

                for layer_key, ldata in entry["features"].items():
                    for feat_id, val in ldata.get("tracked", {}).items():
                        fkey = f"{layer_key}_F{feat_id}"
                        if fkey not in region_summaries[rkey]["features"]:
                            region_summaries[rkey]["features"][fkey] = []
                        region_summaries[rkey]["features"][fkey].append(val)

        # Average the feature lists
        for rkey, rdata in region_summaries.items():
            for fkey, vals in rdata["features"].items():
                if vals:
                    rdata["features"][fkey] = {
                        "mean": round(float(np.mean(vals)), 2),
                        "max": round(float(np.max(vals)), 2),
                        "std": round(float(np.std(vals)), 2),
                        "n": len(vals),
                    }

        # ── Find decision token activations specifically ──
        decision_token_features = {}
        for entry in reversed(tokens_data):
            if entry["region"] == "decision":
                decision_token_features = entry["features"]
                break

        # ── Write per-prompt JSONL record ──
        record = {
            "prompt_id": prompt_idx,
            "condition": p["condition"],
            "tier": p["tier"],
            "is_violation": is_violation,
            "decision": decision,
            "n_tokens": len(tokens_data),
            "generated_text_preview": generated_text[:300],
            "region_summaries": region_summaries,
            "decision_token_features": decision_token_features,
            "tokens": tokens_data,  # full per-token trajectory
        }
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        outfile.flush()

        elapsed = time.time() - t_start
        status = "VIOL" if is_violation else "CLEAN"
        print(f" {status} ({decision}), {len(tokens_data)} tok, {elapsed:.0f}s")

        # Quick feature readout for violations
        if is_violation:
            for rkey in ["reasoning_compliance", "reasoning_non_compliance", "decision"]:
                if rkey in region_summaries:
                    feats = region_summaries[rkey]["features"]
                    parts = []
                    for fk, fv in sorted(feats.items()):
                        if isinstance(fv, dict):
                            parts.append(f"{fk}={fv['mean']:.0f}")
                    if parts:
                        print(f"    {rkey}: {', '.join(parts[:5])}")

        sys.stdout.flush()

    outfile.close()

    # ── Final Analysis ──
    print("\n" + "=" * 60)
    print("FINAL ANALYSIS")
    print(f"Violations: {n_violations}, Clean: {n_clean}")
    print("=" * 60)

    # Reload and analyze
    all_records = []
    with open(outpath, "r") as f:
        for line in f:
            all_records.append(json.loads(line))

    violations = [r for r in all_records if r["is_violation"]]
    cleans = [r for r in all_records if not r["is_violation"]]

    # ── Analysis A: Within violations, compliance vs non-compliance tokens ──
    if violations:
        print("\n[A] WITHIN VIOLATION CASES: compliance-asserting vs other reasoning tokens")
        _compare_subregions(violations, "reasoning_compliance", "reasoning_non_compliance")

    # ── Analysis B: Decision token vs reasoning in violations ──
    if violations:
        print("\n[B] VIOLATION CASES: reasoning tokens vs decision token")
        _compare_region_to_decision(violations)

    # ── Analysis C: Violation vs clean at compliance-asserting positions ──
    if violations and cleans:
        print("\n[C] CROSS-CASE: violation vs clean at compliance-asserting tokens")
        _cross_case_comparison(violations, cleans, "reasoning_compliance")

    # Save summary
    summary_path = outpath.replace(".jsonl", "_analysis.json")
    summary = {
        "n_violations": n_violations,
        "n_clean": n_clean,
        "analysis_a": _get_comparison_data(violations, "reasoning_compliance", "reasoning_non_compliance") if violations else None,
        "analysis_b": _get_decision_comparison_data(violations) if violations else None,
        "analysis_c": _get_cross_case_data(violations, cleans, "reasoning_compliance") if violations and cleans else None,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {outpath}")
    print(f"Saved: {summary_path}")
    print("Done.")


# ─────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────

def _extract_region_features(records, region_key):
    """Extract per-feature means from a specific region across records."""
    feats = {}
    for r in records:
        rs = r.get("region_summaries", {}).get(region_key, {})
        for fkey, fval in rs.get("features", {}).items():
            if isinstance(fval, dict) and "mean" in fval:
                if fkey not in feats:
                    feats[fkey] = []
                feats[fkey].append(fval["mean"])
    return feats


def _compare_subregions(records, region_a, region_b):
    """Compare feature activations between two sub-regions within the same cases."""
    feats_a = _extract_region_features(records, region_a)
    feats_b = _extract_region_features(records, region_b)

    all_keys = sorted(set(list(feats_a.keys()) + list(feats_b.keys())))
    for key in all_keys:
        va = feats_a.get(key, [])
        vb = feats_b.get(key, [])
        if va and vb:
            ma, mb = np.mean(va), np.mean(vb)
            pooled = np.sqrt((np.std(va)**2 + np.std(vb)**2) / 2)
            d = (ma - mb) / pooled if pooled > 0 else 0
            sig = " ***" if abs(d) > 0.8 else (" **" if abs(d) > 0.5 else "")
            print(f"  {key}: {region_a}={ma:.1f}(n={len(va)}) vs "
                  f"{region_b}={mb:.1f}(n={len(vb)}) d={d:.2f}{sig}")


def _compare_region_to_decision(records):
    """Compare reasoning-region activations to decision-token activations."""
    reasoning_feats = _extract_region_features(records, "reasoning")
    decision_feats = {}
    for r in records:
        dtf = r.get("decision_token_features", {})
        for layer_key, ldata in dtf.items():
            for feat_id, val in ldata.get("tracked", {}).items():
                fkey = f"{layer_key}_F{feat_id}"
                if fkey not in decision_feats:
                    decision_feats[fkey] = []
                decision_feats[fkey].append(val)

    all_keys = sorted(set(list(reasoning_feats.keys()) + list(decision_feats.keys())))
    for key in all_keys:
        vr = reasoning_feats.get(key, [])
        vd = decision_feats.get(key, [])
        if vr and vd:
            mr, md = np.mean(vr), np.mean(vd)
            pattern = "SPIKE at decision" if md > mr * 1.5 else (
                "SUSTAINED through reasoning" if abs(mr - md) / max(mr, md, 0.01) < 0.3 else
                "DROPS at decision" if md < mr * 0.5 else "MIXED"
            )
            print(f"  {key}: reasoning={mr:.1f} -> decision={md:.1f} [{pattern}]")


def _cross_case_comparison(violations, cleans, region_key):
    """Compare feature activations at the same region between violation and clean cases."""
    vfeats = _extract_region_features(violations, region_key)
    cfeats = _extract_region_features(cleans, region_key)

    all_keys = sorted(set(list(vfeats.keys()) + list(cfeats.keys())))
    for key in all_keys:
        vv = vfeats.get(key, [])
        cv = cfeats.get(key, [])
        if vv and cv:
            mv, mc = np.mean(vv), np.mean(cv)
            pooled = np.sqrt((np.std(vv)**2 + np.std(cv)**2) / 2)
            d = (mv - mc) / pooled if pooled > 0 else 0
            sig = " ***" if abs(d) > 0.8 else (" **" if abs(d) > 0.5 else "")
            print(f"  {key}: violation={mv:.1f}(n={len(vv)}) vs "
                  f"clean={mc:.1f}(n={len(cv)}) d={d:.2f}{sig}")


def _get_comparison_data(records, region_a, region_b):
    """Return comparison data as dict for JSON export."""
    feats_a = _extract_region_features(records, region_a)
    feats_b = _extract_region_features(records, region_b)
    result = {}
    for key in set(list(feats_a.keys()) + list(feats_b.keys())):
        va, vb = feats_a.get(key, []), feats_b.get(key, [])
        if va and vb:
            pooled = np.sqrt((np.std(va)**2 + np.std(vb)**2) / 2)
            result[key] = {
                f"{region_a}_mean": round(float(np.mean(va)), 2),
                f"{region_b}_mean": round(float(np.mean(vb)), 2),
                "cohens_d": round(float((np.mean(va) - np.mean(vb)) / pooled) if pooled > 0 else 0, 3),
                "n_a": len(va), "n_b": len(vb),
            }
    return result


def _get_decision_comparison_data(records):
    """Return reasoning vs decision comparison as dict."""
    reasoning_feats = _extract_region_features(records, "reasoning")
    decision_feats = {}
    for r in records:
        dtf = r.get("decision_token_features", {})
        for layer_key, ldata in dtf.items():
            for feat_id, val in ldata.get("tracked", {}).items():
                fkey = f"{layer_key}_F{feat_id}"
                if fkey not in decision_feats:
                    decision_feats[fkey] = []
                decision_feats[fkey].append(val)
    result = {}
    for key in set(list(reasoning_feats.keys()) + list(decision_feats.keys())):
        vr, vd = reasoning_feats.get(key, []), decision_feats.get(key, [])
        if vr and vd:
            result[key] = {
                "reasoning_mean": round(float(np.mean(vr)), 2),
                "decision_mean": round(float(np.mean(vd)), 2),
                "pattern": "SPIKE" if np.mean(vd) > np.mean(vr) * 1.5 else (
                    "SUSTAINED" if abs(np.mean(vr) - np.mean(vd)) / max(np.mean(vr), np.mean(vd), 0.01) < 0.3 else
                    "DROP" if np.mean(vd) < np.mean(vr) * 0.5 else "MIXED"
                ),
            }
    return result


def _get_cross_case_data(violations, cleans, region_key):
    """Return cross-case comparison as dict."""
    vfeats = _extract_region_features(violations, region_key)
    cfeats = _extract_region_features(cleans, region_key)
    result = {}
    for key in set(list(vfeats.keys()) + list(cfeats.keys())):
        vv, cv = vfeats.get(key, []), cfeats.get(key, [])
        if vv and cv:
            pooled = np.sqrt((np.std(vv)**2 + np.std(cv)**2) / 2)
            result[key] = {
                "violation_mean": round(float(np.mean(vv)), 2),
                "clean_mean": round(float(np.mean(cv)), 2),
                "cohens_d": round(float((np.mean(vv) - np.mean(cv)) / pooled) if pooled > 0 else 0, 3),
                "n_violation": len(vv), "n_clean": len(cv),
            }
    return result


if __name__ == "__main__":
    main()

 
