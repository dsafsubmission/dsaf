"""
Correspondence Analysis: Per-prompt feature activations by condition and tier.

Computes the four cross-model correspondence numbers:
1. Feature 2996 activation on HRW vs HRW_NA (anchor effect)
2. Features 5785 and 293 activation on HBL vs HOP vs HHP (pressure gradient)
3. Feature 2236 activation on HRW vs HRW_NORM (norm intervention)
4. Feature 2236 activation on strong vs moderate vs qualified tier
"""
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
import json
import time

def main():
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE

    print("=" * 60)
    print("CORRESPONDENCE ANALYSIS")
    print("Per-prompt feature activations by condition and tier")
    print("=" * 60)
    sys.stdout.flush()

    MODEL_ID = "google/gemma-3-27b-it"
    print("\n[1] Loading model...")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        text_backbone = model.model.language_model
    else:
        text_backbone = model.model
    device = next(model.parameters()).device
    print(f"  Loaded")
    sys.stdout.flush()

    print("\n[2] Loading SAE layer 53...")
    sys.stdout.flush()
    sae = SAE.from_pretrained(release="gemma-scope-2-27b-it-res", sae_id="layer_53_width_16k_l0_small")
    sae.eval()
    sae.to(device)

    # Key features
    targets = [2236, 2996, 5785, 293, 838, 870, 170, 14516]

    # Build prompts across conditions
    print("\n[3] Building prompts across conditions...")
    sys.stdout.flush()
    sys.path.insert(0, "/workspace")
    import healthcare

    prompts = []

    # HRW (with anchor)
    nf, g = healthcare.make_prng(42)
    for i in range(30):
        pair = healthcare.generate_pair(nf, g)
        system = healthcare.build_system_prompt()
        up = healthcare.build_user_prompt("HRW", "positive_reward", pair, 21, "twin", False)
        prompts.append({"condition": "HRW", "tier": pair["cert_tier"], "system": system, "prompt": up})

    # HBL (baseline, low pressure)
    nf2, g2 = healthcare.make_prng(99)
    for i in range(20):
        pair = healthcare.generate_pair(nf2, g2)
        system = healthcare.build_system_prompt()
        up = healthcare.build_user_prompt("HBL", "baseline", pair, 21, "twin", False)
        prompts.append({"condition": "HBL", "tier": pair["cert_tier"], "system": system, "prompt": up})

    # HHP (high pressure)
    nf3, g3 = healthcare.make_prng(55)
    for i in range(20):
        pair = healthcare.generate_pair(nf3, g3)
        system = healthcare.build_system_prompt()
        up = healthcare.build_user_prompt("HHP", "high_pressure", pair, 21, "twin", False)
        prompts.append({"condition": "HHP", "tier": pair["cert_tier"], "system": system, "prompt": up})

    # HRW_NA (no anchor)
    nf4, g4 = healthcare.make_prng(77)
    for i in range(20):
        pair = healthcare.generate_pair(nf4, g4)
        system = healthcare.build_system_prompt()
        up = healthcare.build_user_prompt("HRW_NA", "positive_reward", pair, 21, "twin", False, no_anchor=True)
        prompts.append({"condition": "HRW_NA", "tier": pair["cert_tier"], "system": system, "prompt": up})

    # HRW_NORM (with norm prohibition)
    nf5, g5 = healthcare.make_prng(88)
    for i in range(20):
        pair = healthcare.generate_pair(nf5, g5)
        system = healthcare.build_system_prompt(norm_prohibition=True)
        up = healthcare.build_user_prompt("HRW_NORM", "positive_reward", pair, 21, "twin", False)
        prompts.append({"condition": "HRW_NORM", "tier": pair["cert_tier"], "system": system, "prompt": up})

    # HOP (optimization pressure)
    nf6, g6 = healthcare.make_prng(66)
    for i in range(20):
        pair = healthcare.generate_pair(nf6, g6)
        system = healthcare.build_system_prompt()
        up = healthcare.build_user_prompt("HOP", "optimization_pressure", pair, 21, "twin", False)
        prompts.append({"condition": "HOP", "tier": pair["cert_tier"], "system": system, "prompt": up})

    from collections import Counter
    cond_counts = Counter(p["condition"] for p in prompts)
    tier_counts = Counter(p["tier"] for p in prompts)
    print(f"  Built {len(prompts)} prompts: {dict(cond_counts)}")
    print(f"  Tiers: {dict(tier_counts)}")
    sys.stdout.flush()

    # Run all prompts
    print(f"\n[4] Capturing feature activations ({len(prompts)} prompts)...")
    sys.stdout.flush()

    results = []
    for idx, p in enumerate(prompts):
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(prompts)}]")
            sys.stdout.flush()

        messages = [{"role": "user", "content": p["system"] + "\n\n" + p["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        captured = {}
        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["h"] = h[0, -1, :].detach()
        hook = text_backbone.layers[53].register_forward_hook(hook_fn)

        with torch.no_grad():
            model(**inputs)
        hook.remove()

        with torch.no_grad():
            all_acts = sae.encode(captured["h"].unsqueeze(0))[0]

        feat_vals = {str(fi): round(float(all_acts[fi].cpu()), 2) for fi in targets}
        results.append({
            "idx": idx,
            "condition": p["condition"],
            "tier": p["tier"],
            "features": feat_vals,
        })

        del inputs, captured
        torch.cuda.empty_cache()

    # Compute correspondence numbers
    print("\n[5] Computing correspondence numbers...")
    print("=" * 60)
    sys.stdout.flush()

    def mean_feat(entries, fi):
        vals = [e["features"][str(fi)] for e in entries]
        return np.mean(vals) if vals else 0

    # 1. Feature 2996 on HRW vs HRW_NA (anchor effect)
    hrw = [r for r in results if r["condition"] == "HRW"]
    hrw_na = [r for r in results if r["condition"] == "HRW_NA"]
    f2996_hrw = mean_feat(hrw, 2996)
    f2996_na = mean_feat(hrw_na, 2996)
    print(f"\n  1. ANCHOR EFFECT (Feature 2996 - 'average/deny')")
    print(f"     HRW (with anchor):  {f2996_hrw:.1f}")
    print(f"     HRW_NA (no anchor): {f2996_na:.1f}")
    print(f"     Difference: {f2996_hrw - f2996_na:.1f}")

    # 2. Features 5785 and 293 on HBL vs HOP vs HHP (pressure gradient)
    hbl = [r for r in results if r["condition"] == "HBL"]
    hop = [r for r in results if r["condition"] == "HOP"]
    hhp = [r for r in results if r["condition"] == "HHP"]
    print(f"\n  2. PRESSURE GRADIENT")
    for fi, name in [(5785, "OPPORT/financial"), (293, "COMP/compliance")]:
        v_hbl = mean_feat(hbl, fi)
        v_hop = mean_feat(hop, fi)
        v_hhp = mean_feat(hhp, fi)
        print(f"     Feature {fi} ({name}): HBL={v_hbl:.1f} → HOP={v_hop:.1f} → HHP={v_hhp:.1f}")
        monotonic = v_hbl <= v_hop <= v_hhp
        print(f"     Monotonic increase: {'YES' if monotonic else 'NO'}")

    # 3. Feature 2236 on HRW vs HRW_NORM (norm intervention)
    hrw_norm = [r for r in results if r["condition"] == "HRW_NORM"]
    f2236_hrw = mean_feat(hrw, 2236)
    f2236_norm = mean_feat(hrw_norm, 2236)
    print(f"\n  3. NORM INTERVENTION (Feature 2236 - 'NOT/LOS/must')")
    print(f"     HRW (no norm text):  {f2236_hrw:.1f}")
    print(f"     HRW_NORM (with norm): {f2236_norm:.1f}")
    print(f"     Difference: {f2236_norm - f2236_hrw:.1f}")

    # 4. Feature 2236 by tier
    strong = [r for r in results if r["tier"] == "strong"]
    moderate = [r for r in results if r["tier"] == "moderate"]
    qualified = [r for r in results if r["tier"] == "qualified"]
    f2236_s = mean_feat(strong, 2236)
    f2236_m = mean_feat(moderate, 2236)
    f2236_q = mean_feat(qualified, 2236)
    print(f"\n  4. TIER EFFECT (Feature 2236)")
    print(f"     Strong:    {f2236_s:.1f} (violation rate: 12%)")
    print(f"     Moderate:  {f2236_m:.1f} (violation rate: 80%)")
    print(f"     Qualified: {f2236_q:.1f} (violation rate: 100%)")
    print(f"     Strong > Moderate > Qualified: {'YES' if f2236_s > f2236_m > f2236_q else 'NO'}")

    # Also compute all target features for completeness
    print(f"\n  ALL FEATURES BY CONDITION:")
    for fi in targets:
        print(f"    Feature {fi}:")
        for cond in ["HBL", "HRW", "HOP", "HHP", "HRW_NA", "HRW_NORM"]:
            entries = [r for r in results if r["condition"] == cond]
            if entries:
                v = mean_feat(entries, fi)
                print(f"      {cond:10s}: {v:.1f}")

    # Save
    output = {
        "anchor_effect": {"feature": 2996, "HRW": round(f2996_hrw, 2), "HRW_NA": round(f2996_na, 2)},
        "pressure_gradient": {
            "feature_5785": {"HBL": round(mean_feat(hbl, 5785), 2), "HOP": round(mean_feat(hop, 5785), 2), "HHP": round(mean_feat(hhp, 5785), 2)},
            "feature_293": {"HBL": round(mean_feat(hbl, 293), 2), "HOP": round(mean_feat(hop, 293), 2), "HHP": round(mean_feat(hhp, 293), 2)},
        },
        "norm_intervention": {"feature": 2236, "HRW": round(f2236_hrw, 2), "HRW_NORM": round(f2236_norm, 2)},
        "tier_effect": {"feature": 2236, "strong": round(f2236_s, 2), "moderate": round(f2236_m, 2), "qualified": round(f2236_q, 2)},
        "per_prompt_data": results,
    }
    with open("/root/sae_correspondence.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to /root/sae_correspondence.json")
    print("DONE")

if __name__ == "__main__":
    main()

 
