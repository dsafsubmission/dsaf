"""
DEFINITIVE CORPUS COUNT.

Rules (agreed with user, final, do not change):
1. Include ALL seeds (42, 43, 44, 45, 46, 47, etc.) -- different patients, unique decisions
2. Include ALL temperatures (0.0, 0.3, 0.7, 1.0) -- different outputs, unique decisions
3. EXCLUDE superseded pilots: if same (model, domain, condition, seed, temp) has both
   a small run (N < 40% of largest) and a large run (N >= 200), the small one is a pilot
4. EXCLUDE fiduciary domain -- not part of study
5. EXCLUDE Unknown model and Llama 3.1 8B -- not study models
6. EXCLUDE files in backup/stubs folders
7. EXCLUDE merged_ files (duplicates of dsaf_ files)

This script is the SINGLE SOURCE OF TRUTH. Run it and the number it prints
is the number that goes in the paper. Period.
"""
import glob, csv, os, re
from collections import defaultdict

def count():
    # Catalog all files
    files = []
    for f in sorted(glob.glob("results/*decisions*.csv")):
        bn = os.path.basename(f)
        if bn.startswith("merged_") or "backup" in f or "stubs" in f or "fiduciary" in f:
            continue

        # Determine seed from meta.json (the ONLY authoritative source)
        import json as _json
        file_seed = 42  # default only if no meta exists
        ts_match = re.search(r"(\d{8}_\d{6})", bn)
        if ts_match:
            for meta_candidate in glob.glob(os.path.join(os.path.dirname(f), "*" + ts_match.group(1) + "*meta.json")):
                try:
                    with open(meta_candidate, "r") as mf:
                        meta = _json.loads(mf.read())
                    if meta.get("patient_seed") is not None:
                        file_seed = meta["patient_seed"]
                        break
                except:
                    pass
        else:
            # Old-format files without timestamp: check filename for seed
            seed_in_name = re.search(r"seed(\d+)", bn)
            if seed_in_name:
                file_seed = int(seed_in_name.group(1))

        if "healthcare" in bn: domain = "healthcare"
        elif "lending" in bn: domain = "lending"
        elif "trading" in bn: domain = "trading"
        elif "cancer" in bn: domain = "cancer"
        else: continue

        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                row = next(reader)
                model_id = row.get("model_id", "")
                test_id = row.get("test_id", "")
                temp = row.get("temperature", "0.3") or "0.3"
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                n = sum(1 for _ in csv.DictReader(fh))
        except:
            continue

        if not model_id or not test_id:
            continue
        if "llama-3.1-8b" in model_id.lower():
            continue

        files.append({
            "path": f, "bn": bn, "model": model_id, "domain": domain,
            "test_id": test_id, "seed": file_seed, "temp": temp, "n": n
        })

    # Find superseded pilots
    groups = defaultdict(list)
    for fi in files:
        key = (fi["model"], fi["domain"], fi["test_id"], fi["seed"], fi["temp"])
        groups[key].append(fi)

    legitimate = []
    superseded = []

    for key, group_files in groups.items():
        if len(group_files) == 1:
            legitimate.append(group_files[0])
            continue

        group_files.sort(key=lambda x: x["n"], reverse=True)
        largest = group_files[0]["n"]

        for fi in group_files:
            if fi["n"] < largest * 0.4 and largest >= 200:
                superseded.append(fi)
            else:
                legitimate.append(fi)

    total = sum(f["n"] for f in legitimate)

    # Breakdowns
    by_model = defaultdict(int)
    by_domain = defaultdict(int)
    by_seed = defaultdict(int)
    by_temp = defaultdict(int)
    for f in legitimate:
        # Map model_id to display name
        mid = f["model"]
        if "claude-sonnet" in mid: name = "Claude Sonnet 4"
        elif "gpt-4o" in mid: name = "GPT-4o"
        elif "deepseek" in mid: name = "DeepSeek V3"
        elif "gemini" in mid or "google" in mid: name = "Gemini 2.5 Pro"
        elif "qwen" in mid: name = "Qwen 2.5-72B"
        elif "llama-4" in mid or "maverick" in mid: name = "Llama 4 Maverick"
        elif "gemma" in mid: name = "Gemma 3 27B"
        elif "llama-3.3" in mid: name = "Llama 3.3 70B"
        elif "opus" in mid: name = "Claude Opus 4"
        elif "haiku" in mid: name = "Claude Haiku"
        else: name = mid
        by_model[name] += f["n"]
        by_domain[f["domain"]] += f["n"]
        by_seed[f["seed"]] += f["n"]
        by_temp[f["temp"]] += f["n"]

    print("=" * 60)
    print("DEFINITIVE CORPUS COUNT")
    print("=" * 60)
    print(f"\nTOTAL LEGITIMATE DECISIONS: {total:,}")
    print(f"Legitimate files: {len(legitimate)}")
    print(f"Superseded pilots removed: {len(superseded)} ({sum(f['n'] for f in superseded):,} decisions)")
    print(f"\nBy model:")
    for name, n in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"  {name}: {n:,}")
    print(f"\nBy domain:")
    for d, n in sorted(by_domain.items(), key=lambda x: -x[1]):
        print(f"  {d}: {n:,}")
    print(f"\nBy seed:")
    for s, n in sorted(by_seed.items()):
        print(f"  Seed {s}: {n:,}")
    print(f"\nBy temperature:")
    for t, n in sorted(by_temp.items()):
        print(f"  Temp {t}: {n:,}")
    print(f"\n{'=' * 60}")
    print(f"FINAL NUMBER: {total:,}")
    print(f"Over 200K: {total > 200000}")
    print(f"{'=' * 60}")

    return total, legitimate, superseded

if __name__ == "__main__":
    count()
