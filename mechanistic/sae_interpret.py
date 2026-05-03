"""
SAE Feature Interpretation: What do the top compliance-fabrication features represent?

Two methods:
1. Logit lens: Project SAE decoder vectors through the unembedding matrix
   to see what tokens each feature promotes in output space.
2. Max-activating tokens: Find which input tokens most strongly activate
   each feature across our healthcare prompts.
"""
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
import json
import time
import gc

def main():
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE

    print("=" * 60)
    print("SAE FEATURE INTERPRETATION")
    print("What do the top violation-predictive features represent?")
    print("=" * 60)
    sys.stdout.flush()

    MODEL_ID = "google/gemma-3-27b-it"

    # ── Load tokenizer and model ──
    print("\n[1] Loading model...")
    sys.stdout.flush()
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
        raise RuntimeError("Cannot find text backbone")

    total_layers = len(text_backbone.layers)
    device = next(model.parameters()).device
    print(f"  Loaded, {total_layers} layers")
    sys.stdout.flush()

    # ── Get unembedding matrix ──
    # For logit lens: project decoder vectors through the LM head
    if hasattr(model, 'lm_head'):
        unembed = model.lm_head.weight.data.float()  # (vocab, d_model)
    else:
        unembed = model.get_output_embeddings().weight.data.float()
    vocab_size, d_model = unembed.shape
    print(f"  Unembedding: {unembed.shape}")
    sys.stdout.flush()

    # ── Load SAEs ──
    print("\n[2] Loading SAEs...")
    sys.stdout.flush()
    SAE_RELEASE = "gemma-scope-2-27b-it-res"

    # Top features from our analysis (from sae_results.json)
    # Layer 31 top violation+ features
    layer31_viol_features = [6095, 1382, 28, 3205, 7648, 5785, 3269, 6649, 373, 3501, 184]
    # Layer 31 top clean+ features
    layer31_clean_features = [635, 7105, 14861, 2688, 8993, 119, 2865]
    # Layer 53 top violation+ features
    layer53_viol_features = [170, 14516, 9305, 13310, 4745, 293, 3541, 4003, 7347, 9786, 1687, 2107, 14038, 4041, 2996]
    # Layer 53 top clean+ features
    layer53_clean_features = [838, 870, 2236]

    target_layers = {
        31: {"viol": layer31_viol_features, "clean": layer31_clean_features},
        53: {"viol": layer53_viol_features, "clean": layer53_clean_features},
    }

    sae_dict = {}
    for layer_idx in target_layers:
        sae = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=f"layer_{layer_idx}_width_16k_l0_small",
        )
        sae.eval()
        sae_dict[layer_idx] = sae
        print(f"  Layer {layer_idx}: {sae.cfg.d_sae} features")
    sys.stdout.flush()

    # ── METHOD 1: Logit Lens ──
    # Project each feature's decoder vector through unembedding to see
    # what output tokens it promotes
    print("\n[3] Logit Lens — what tokens does each feature promote?")
    print("=" * 60)
    sys.stdout.flush()

    interpretation_results = {}

    for layer_idx, sae in sae_dict.items():
        # W_dec: (n_features, d_model) — each row is a feature's direction
        W_dec = sae.W_dec.data.float().cpu()  # (16384, d_model)

        print(f"\n  === LAYER {layer_idx} ===")
        sys.stdout.flush()

        layer_results = {}

        for category, feature_list in target_layers[layer_idx].items():
            print(f"\n  --- {category.upper()} features ---")
            for fi in feature_list:
                # Get this feature's decoder direction
                feat_dir = W_dec[fi]  # (d_model,)

                # Project through unembedding: which tokens does this direction point toward?
                logits = unembed.cpu() @ feat_dir  # (vocab,)

                # Top promoted tokens
                top_k = 20
                top_indices = torch.topk(logits, top_k).indices
                top_tokens = [tokenizer.decode([idx.item()]).strip() for idx in top_indices]
                top_scores = [round(logits[idx].item(), 2) for idx in top_indices]

                # Bottom tokens (what it suppresses)
                bot_indices = torch.topk(-logits, 10).indices
                bot_tokens = [tokenizer.decode([idx.item()]).strip() for idx in bot_indices]

                print(f"\n  Feature {fi} ({category}):")
                print(f"    PROMOTES: {', '.join(top_tokens[:15])}")
                print(f"    SUPPRESSES: {', '.join(bot_tokens[:10])}")
                sys.stdout.flush()

                layer_results[f"feature_{fi}"] = {
                    "feature_index": fi,
                    "category": category,
                    "top_promoted_tokens": list(zip(top_tokens, top_scores)),
                    "top_suppressed_tokens": bot_tokens[:10],
                }

        interpretation_results[f"layer_{layer_idx}_logit_lens"] = layer_results

    # ── METHOD 2: Max-activating tokens from our prompts ──
    print("\n\n[4] Max-activating tokens from healthcare prompts")
    print("=" * 60)
    sys.stdout.flush()

    sys.path.insert(0, "/workspace")
    import healthcare

    system_prompt = healthcare.build_system_prompt()
    nf, g = healthcare.make_prng(42)

    # Build a smaller set of prompts for interpretation
    prompts = []
    for i in range(100):
        pair = healthcare.generate_pair(nf, g)
        up = healthcare.build_user_prompt("HRW", "positive_reward", pair, 21, "twin", False)
        prompts.append(up)
        if len(prompts) >= 15:
            break

    nf2, g2 = healthcare.make_prng(99)
    for i in range(100):
        pair = healthcare.generate_pair(nf2, g2)
        up = healthcare.build_user_prompt("HBL", "baseline", pair, 21, "twin", False)
        prompts.append(up)
        if len(prompts) >= 25:
            break

    print(f"  Using {len(prompts)} prompts for activation analysis")
    sys.stdout.flush()

    # For each prompt, run forward pass and record per-token SAE activations
    # for our target features
    all_target_features = set()
    for layer_idx in target_layers:
        for cat_features in target_layers[layer_idx].values():
            all_target_features.update(cat_features)

    max_activations = {fi: [] for fi in all_target_features}  # fi -> [(token_str, activation, context)]

    for pidx, prompt_text in enumerate(prompts):
        if (pidx + 1) % 5 == 0:
            print(f"  [{pidx+1}/{len(prompts)}]")
            sys.stdout.flush()

        messages = [
            {"role": "user", "content": system_prompt + "\n\n" + prompt_text},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"][0]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Capture ALL token activations (not just last token)
        captured = {}

        def make_hook(layer_idx):
            def hook_fn(module, inp, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                captured[layer_idx] = h[0].detach()  # (seq_len, d_model)
            return hook_fn

        hooks = []
        for layer_idx in sae_dict:
            hook = text_backbone.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        # Encode through SAE and find max-activating tokens
        for layer_idx, sae in sae_dict.items():
            if layer_idx not in captured:
                continue
            h = captured[layer_idx]  # (seq_len, d_model)
            with torch.no_grad():
                sae_on_device = sae.to(h.device)
                feature_acts = sae_on_device.encode(h)  # (seq_len, n_features)

            seq_len = feature_acts.shape[0]
            tokens = [tokenizer.decode([input_ids[t].item()]) for t in range(min(seq_len, len(input_ids)))]

            for fi in all_target_features:
                acts = feature_acts[:, fi].cpu().float().numpy()
                # Find top 3 activating positions in this prompt
                top_positions = np.argsort(acts)[-3:][::-1]
                for pos in top_positions:
                    if pos < len(tokens) and acts[pos] > 0:
                        # Get surrounding context (5 tokens before and after)
                        start = max(0, pos - 5)
                        end = min(len(tokens), pos + 6)
                        context = "".join(tokens[start:end])
                        max_activations[fi].append({
                            "token": tokens[pos].strip(),
                            "activation": round(float(acts[pos]), 2),
                            "context": context.strip(),
                            "position": int(pos),
                            "prompt_idx": pidx,
                        })

        del inputs, captured
        torch.cuda.empty_cache()

    # Sort and report top activating tokens per feature
    print("\n\n  MAX-ACTIVATING TOKENS PER FEATURE:")
    print("=" * 60)
    sys.stdout.flush()

    activation_results = {}

    for layer_idx in target_layers:
        print(f"\n  === LAYER {layer_idx} ===")
        for category, feature_list in target_layers[layer_idx].items():
            print(f"\n  --- {category.upper()} features ---")
            for fi in feature_list:
                entries = sorted(max_activations[fi], key=lambda x: -x["activation"])
                unique_tokens = {}
                for e in entries:
                    tok = e["token"]
                    if tok not in unique_tokens:
                        unique_tokens[tok] = e
                    if len(unique_tokens) >= 10:
                        break

                top_entries = list(unique_tokens.values())[:10]
                token_summary = ", ".join(f"'{e['token']}'({e['activation']})" for e in top_entries[:8])
                print(f"\n  Feature {fi} ({category}):")
                print(f"    Top tokens: {token_summary}")
                if top_entries:
                    print(f"    Example context: ...{top_entries[0]['context']}...")
                    if len(top_entries) > 1:
                        print(f"    Example context: ...{top_entries[1]['context']}...")
                sys.stdout.flush()

                activation_results[f"layer_{layer_idx}_feature_{fi}"] = {
                    "feature_index": fi,
                    "layer": layer_idx,
                    "category": category,
                    "top_activating_tokens": top_entries[:10],
                }

    # ── Save results ──
    print("\n\n[5] Saving interpretation results...")
    results = {
        "logit_lens": interpretation_results,
        "max_activating_tokens": activation_results,
        "metadata": {
            "model": MODEL_ID,
            "sae_release": SAE_RELEASE,
            "n_prompts_for_activation": len(prompts),
            "method": "logit_lens + max_activating_tokens",
        }
    }

    with open("/root/sae_interpretation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  Saved to /root/sae_interpretation.json")
    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()
