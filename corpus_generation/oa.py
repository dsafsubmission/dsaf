"""
oa.py — OA Adversarial Testing Framework: main entry point.

Usage examples:
  python oa.py --model ollama/qwen2.5:14b --domain trading --test A
  python oa.py --model ollama/qwen2.5:14b --domain trading --test all
  python oa.py --model ollama/qwen2.5:14b --domain lending --test L1,F1
  python oa.py --model ollama/qwen2.5:14b --domain all --test all
  python oa.py --model deepseek/deepseek-v3 --domain trading --test all --api-key sk-...
  python oa.py --model ollama/qwen2.5:14b --domain trading --test A --quick
"""

import io
import sys
# Force UTF-8 output on Windows (handles box-drawing chars in print statements)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import argparse
import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone

from llm_client import LLMClient, MODELS, TEMPERATURE
import trading
import trading_v2
import lending
import healthcare
import cancer
import fiduciary
import pension

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Quick mode runs 20 decisions per test instead of full counts — for debugging only
QUICK_DECISIONS = 20


def main():
    parser = argparse.ArgumentParser(
        description="OA Adversarial Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True,
        help=f"Model ID. Options: {list(MODELS.keys())}"
    )
    parser.add_argument(
        "--domain", default="trading",
        help="Domain: trading | lending | all"
    )
    parser.add_argument(
        "--test", default="all",
        help="Test ID(s): e.g. A, A,C,F, all"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key (also reads from DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=f"Quick mode: run only {QUICK_DECISIONS} decisions per test (for debugging)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for lending applicant generator (default: 42)"
    )
    args = parser.parse_args()

    # ── Resolve API key ───────────────────────────────────────────
    api_key = args.api_key
    if not api_key:
        model_prefix = args.model.split("/")[0]
        env_map = {
            "deepseek":  "DEEPSEEK_API_KEY",
            "openai":    "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "ollama":    None,
        }
        env_var = env_map.get(model_prefix)
        if env_var:
            api_key = os.environ.get(env_var, "").strip()
            if not api_key:
                print(f"ERROR: Set {env_var} environment variable or pass --api-key")
                sys.exit(1)

    # ── Build LLM client ──────────────────────────────────────────
    try:
        client = LLMClient(args.model, api_key)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Verify Ollama is reachable
    if args.model.startswith("ollama/"):
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:11434", timeout=3)
        except Exception:
            print("ERROR: Ollama is not running.")
            print("  Start it with: ollama serve")
            sys.exit(1)

    # ── Determine domains and tests ───────────────────────────────
    domain_arg = args.domain.lower()
    domains = (["trading", "trading2", "lending", "healthcare", "cancer", "fiduciary", "pension"]
               if domain_arg == "all" else [domain_arg])

    test_arg = args.test.lower()

    def resolve_tests(domain: str) -> list[str]:
        domain_tests = {
            "trading":    list(trading.TESTS.keys()),
            "trading2":   list(trading_v2.TESTS.keys()),
            "lending":    list(lending.TESTS.keys()),
            "healthcare": list(healthcare.TESTS.keys()),
            "cancer":     list(cancer.TESTS.keys()),
            "fiduciary":  list(fiduciary.TESTS.keys()),
            "pension":    list(pension.TESTS.keys()),
        }
        available = domain_tests[domain]
        if test_arg == "all":
            return available
        requested = [t.strip().upper() for t in args.test.split(",")]
        valid = [t for t in requested if t in available]
        if not valid:
            print(f"WARNING: No valid tests for domain={domain} in '{args.test}'")
        return valid

    # ── Quick-mode decision override ──────────────────────────────
    # Proportionally compress each regime so ALL regimes are covered,
    # giving at least QUICK_DECISIONS_PER_REGIME decisions per regime.
    QUICK_DECISIONS_PER_REGIME = 5
    if args.quick:
        all_test_dicts = [trading.TESTS, trading_v2.TESTS, lending.TESTS]
        for tests_dict in all_test_dicts:
            for t in tests_dict.values():
                t["_orig_total"]   = t["total_decisions"]
                t["_orig_regimes"] = list(t["regimes"])
                new_regimes = []
                cursor = 0
                for rname, rstart, rend in t["regimes"]:
                    size = max(QUICK_DECISIONS_PER_REGIME, rend - rstart)
                    # Scale proportionally but enforce minimum
                    orig_size  = rend - rstart
                    orig_total = t["total_decisions"]
                    new_size   = max(
                        QUICK_DECISIONS_PER_REGIME,
                        round(QUICK_DECISIONS * orig_size / orig_total)
                    )
                    new_regimes.append((rname, cursor, cursor + new_size))
                    cursor += new_size
                t["regimes"]         = new_regimes
                t["total_decisions"] = cursor
        n_regimes = max(len(t["regimes"]) for d in all_test_dicts for t in d.values())
        print(f"\n[QUICK MODE] ~{QUICK_DECISIONS_PER_REGIME} decisions/regime, all regimes covered")

    # ── Set up output files ───────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id   = str(uuid.uuid4())[:8]
    model_safe = args.model.replace("/", "_").replace(":", "-")

    # Metadata file
    from llm_client import SEED as LLM_SEED
    meta = {
        "run_id":      run_id,
        "timestamp":   ts,
        "model":       args.model,
        "temperature": TEMPERATURE,
        "llm_seed":    LLM_SEED,
        "patient_seed": args.seed,
        "quick_mode":  args.quick,
        "domains":     domains,
        "tests":       {d: resolve_tests(d) for d in domains},
    }
    meta_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nRun ID:     {run_id}")
    print(f"Model:      {args.model}")
    print(f"Temp:       {TEMPERATURE}  |  LLM seed: {LLM_SEED}")
    print(f"Patient seed: {args.seed}")
    print(f"Output dir: {RESULTS_DIR}")

    all_stats    = []
    all_violations = []

    # ── Trading ───────────────────────────────────────────────────
    if "trading" in domains:
        test_ids = resolve_tests("trading")
        if test_ids:
            # Load price data
            needed = set()
            for tid in test_ids:
                for rname, _, _ in trading.TESTS[tid]["regimes"]:
                    needed.add(rname)
            print(f"\nLoading trading price data...")
            regime_cache = {}
            for rname in needed:
                regime_cache[rname] = trading.load_regime_ticks(rname)
                print(f"  {rname}: {len(regime_cache[rname])} ticks")

            dec_path  = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_trading_decisions.csv")
            viol_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_trading_violations.csv")
            violations_log = []

            with open(dec_path, "w", newline="", encoding="utf-8") as dec_f:
                dec_w = csv.DictWriter(dec_f, fieldnames=trading.DECISIONS_FIELDS)
                dec_w.writeheader()
                for tid in test_ids:
                    stats = trading.run_test(
                        tid, client, regime_cache,
                        dec_w, dec_f, violations_log, run_id)
                    stats["domain"] = "trading"
                    all_stats.append(stats)

            # Write violations (with realized premiums filled in)
            with open(viol_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=trading.VIOLATIONS_FIELDS)
                w.writeheader()
                for v in violations_log:
                    w.writerow({k: v.get(k, "") for k in trading.VIOLATIONS_FIELDS})

            print(f"\n  Trading decisions -> {dec_path}")
            print(f"  Trading violations -> {viol_path}")
            all_violations.extend(violations_log)

    # ── Trading V2 (matched-pair) ────────────────────────────────
    if "trading2" in domains:
        test_ids = resolve_tests("trading2")
        if test_ids:
            tid_suffix = "_".join(test_ids)
            dec_path  = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_trading2_decisions.csv")
            viol_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_trading2_violations.csv")
            mcnm_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_trading2_mcnemar.csv")
            violations_log = []
            mcnemar_log = []

            with open(dec_path, "w", newline="", encoding="utf-8") as dec_f:
                dec_w = csv.DictWriter(dec_f, fieldnames=trading_v2.DECISIONS_FIELDS)
                dec_w.writeheader()
                for tid in test_ids:
                    stats = trading_v2.run_test(
                        tid, client, dec_w, dec_f,
                        violations_log, mcnemar_log, run_id, seed=args.seed)
                    stats["domain"] = "trading2"
                    all_stats.append(stats)

            with open(viol_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=trading_v2.VIOLATIONS_FIELDS)
                w.writeheader()
                for v in violations_log:
                    w.writerow({k: v.get(k, "") for k in trading_v2.VIOLATIONS_FIELDS})

            with open(mcnm_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=trading_v2.MCNEMAR_FIELDS)
                w.writeheader()
                for v in mcnemar_log:
                    w.writerow({k: v.get(k, "") for k in trading_v2.MCNEMAR_FIELDS})

            print(f"\n  Trading2 decisions  -> {dec_path}")
            print(f"  Trading2 violations -> {viol_path}")
            print(f"  Trading2 McNemar    -> {mcnm_path}")
            all_violations.extend(violations_log)

    # ── Lending ───────────────────────────────────────────────────
    if "lending" in domains:
        test_ids = resolve_tests("lending")
        if test_ids:
            # Include test IDs in filename to prevent parallel processes from
            # colliding on the same file when launched within the same second.
            tid_suffix = "_".join(test_ids)
            dec_path   = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_lending_decisions.csv")
            viol_path  = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_lending_violations.csv")
            mcnm_path  = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_lending_mcnemar.csv")
            violations_log = []
            mcnemar_log    = []

            with open(dec_path, "w", newline="", encoding="utf-8") as dec_f:
                dec_w = csv.DictWriter(dec_f, fieldnames=lending.DECISIONS_FIELDS)
                dec_w.writeheader()
                for tid in test_ids:
                    stats = lending.run_test(
                        tid, client,
                        dec_w, dec_f,
                        violations_log, mcnemar_log,
                        run_id, seed=args.seed)
                    stats["domain"] = "lending"
                    all_stats.append(stats)

            with open(viol_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=lending.VIOLATIONS_FIELDS)
                w.writeheader()
                for v in violations_log:
                    w.writerow({k: v.get(k, "") for k in lending.VIOLATIONS_FIELDS})

            with open(mcnm_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=lending.MCNEMAR_FIELDS)
                w.writeheader()
                for v in mcnemar_log:
                    w.writerow({k: v.get(k, "") for k in lending.MCNEMAR_FIELDS})

            print(f"\n  Lending decisions  -> {dec_path}")
            print(f"  Lending violations -> {viol_path}")
            print(f"  Lending McNemar    -> {mcnm_path}")
            all_violations.extend(violations_log)

    # ── Healthcare ────────────────────────────────────────────────
    if "healthcare" in domains:
        test_ids = resolve_tests("healthcare")
        if test_ids:
            tid_suffix = "_".join(test_ids)
            dec_path  = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_healthcare_decisions.csv")
            viol_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_healthcare_violations.csv")
            mcnm_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_healthcare_mcnemar.csv")
            violations_log = []
            mcnemar_log = []

            with open(dec_path, "w", newline="", encoding="utf-8") as dec_f:
                dec_w = csv.DictWriter(dec_f, fieldnames=healthcare.DECISIONS_FIELDS)
                dec_w.writeheader()
                for tid in test_ids:
                    stats = healthcare.run_test(
                        tid, client, dec_w, dec_f,
                        violations_log, mcnemar_log, run_id, seed=args.seed)
                    stats["domain"] = "healthcare"
                    all_stats.append(stats)

            with open(viol_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=healthcare.VIOLATIONS_FIELDS)
                w.writeheader()
                for v in violations_log:
                    w.writerow({k: v.get(k, "") for k in healthcare.VIOLATIONS_FIELDS})

            with open(mcnm_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=healthcare.MCNEMAR_FIELDS)
                w.writeheader()
                for v in mcnemar_log:
                    w.writerow({k: v.get(k, "") for k in healthcare.MCNEMAR_FIELDS})

            print(f"\n  Healthcare decisions  -> {dec_path}")
            print(f"  Healthcare violations -> {viol_path}")
            print(f"  Healthcare McNemar    -> {mcnm_path}")
            all_violations.extend(violations_log)

    # ── Cancer ────────────────────────────────────────────────────
    if "cancer" in domains:
        test_ids = resolve_tests("cancer")
        if test_ids:
            tid_suffix = "_".join(test_ids)
            dec_path  = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_cancer_decisions.csv")
            viol_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_cancer_violations.csv")
            mcnm_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_cancer_mcnemar.csv")
            violations_log = []
            mcnemar_log = []

            with open(dec_path, "w", newline="", encoding="utf-8") as dec_f:
                dec_w = csv.DictWriter(dec_f, fieldnames=cancer.DECISIONS_FIELDS)
                dec_w.writeheader()
                for tid in test_ids:
                    stats = cancer.run_test(
                        tid, client, dec_w, dec_f,
                        violations_log, mcnemar_log, run_id, seed=args.seed)
                    stats["domain"] = "cancer"
                    all_stats.append(stats)

            with open(viol_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cancer.VIOLATIONS_FIELDS)
                w.writeheader()
                for v in violations_log:
                    w.writerow({k: v.get(k, "") for k in cancer.VIOLATIONS_FIELDS})

            with open(mcnm_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cancer.MCNEMAR_FIELDS)
                w.writeheader()
                for v in mcnemar_log:
                    w.writerow({k: v.get(k, "") for k in cancer.MCNEMAR_FIELDS})

            print(f"\n  Cancer decisions  -> {dec_path}")
            print(f"  Cancer violations -> {viol_path}")
            print(f"  Cancer McNemar    -> {mcnm_path}")
            all_violations.extend(violations_log)

    # ── Fiduciary ────────────────────────────────────────────────
    if "fiduciary" in domains:
        test_ids = resolve_tests("fiduciary")
        if test_ids:
            tid_suffix = "_".join(test_ids)
            dec_path  = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_fiduciary_decisions.csv")
            viol_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_fiduciary_violations.csv")
            mcnm_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_fiduciary_mcnemar.csv")
            violations_log = []
            mcnemar_log = []

            with open(dec_path, "w", newline="", encoding="utf-8") as dec_f:
                dec_w = csv.DictWriter(dec_f, fieldnames=fiduciary.DECISIONS_FIELDS)
                dec_w.writeheader()
                for tid in test_ids:
                    stats = fiduciary.run_test(
                        tid, client, dec_w, dec_f,
                        violations_log, mcnemar_log, run_id, seed=args.seed)
                    stats["domain"] = "fiduciary"
                    all_stats.append(stats)

            with open(viol_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fiduciary.VIOLATIONS_FIELDS)
                w.writeheader()
                for v in violations_log:
                    w.writerow({k: v.get(k, "") for k in fiduciary.VIOLATIONS_FIELDS})

            with open(mcnm_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fiduciary.MCNEMAR_FIELDS)
                w.writeheader()
                for v in mcnemar_log:
                    w.writerow({k: v.get(k, "") for k in fiduciary.MCNEMAR_FIELDS})

            print(f"\n  Fiduciary decisions  -> {dec_path}")
            print(f"  Fiduciary violations -> {viol_path}")
            print(f"  Fiduciary McNemar    -> {mcnm_path}")
            all_violations.extend(violations_log)

    # ── Pension ──────────────────────────────────────────────────
    if "pension" in domains:
        test_ids = resolve_tests("pension")
        if test_ids:
            tid_suffix = "_".join(test_ids)
            dec_path  = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_pension_decisions.csv")
            viol_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_pension_violations.csv")
            mcnm_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_{tid_suffix}_pension_mcnemar.csv")
            violations_log = []
            mcnemar_log = []

            with open(dec_path, "w", newline="", encoding="utf-8") as dec_f:
                dec_w = csv.DictWriter(dec_f, fieldnames=pension.DECISIONS_FIELDS)
                dec_w.writeheader()
                for tid in test_ids:
                    stats = pension.run_test(
                        tid, client, dec_w, dec_f,
                        violations_log, mcnemar_log, run_id, seed=args.seed)
                    stats["domain"] = "pension"
                    all_stats.append(stats)

            with open(viol_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=pension.VIOLATIONS_FIELDS)
                w.writeheader()
                for v in violations_log:
                    w.writerow({k: v.get(k, "") for k in pension.VIOLATIONS_FIELDS})

            with open(mcnm_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=pension.MCNEMAR_FIELDS)
                w.writeheader()
                for v in mcnemar_log:
                    w.writerow({k: v.get(k, "") for k in pension.MCNEMAR_FIELDS})

            print(f"\n  Pension decisions  -> {dec_path}")
            print(f"  Pension violations -> {viol_path}")
            print(f"  Pension McNemar    -> {mcnm_path}")
            all_violations.extend(violations_log)

    # Restore quick-mode overrides
    if args.quick:
        for t in {**trading.TESTS, **trading_v2.TESTS, **lending.TESTS, **healthcare.TESTS, **cancer.TESTS, **fiduciary.TESTS, **pension.TESTS}.values():
            if "_orig_total" in t:
                t["total_decisions"] = t.pop("_orig_total")
            if "_orig_regimes" in t:
                t["regimes"] = t.pop("_orig_regimes")

    # ── Summary ───────────────────────────────────────────────────
    stats_path = os.path.join(RESULTS_DIR, f"oa_{model_safe}_{ts}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY  |  Model: {args.model}  |  Run: {run_id}")
    print(f"{'='*70}")
    print(f"  {'Test':<8} {'Domain':<12} {'ViolRate':>9} {'AsymRatio':>11} "
          f"{'p-value':>10} {'h':>7} {'OR':>8}")
    print(f"  {'-'*66}")
    for s in all_stats:
        print(
            f"  {s['test_id']:<8} "
            f"{s.get('domain',''):<12} "
            f"{s['total_vrate']*100:>8.2f}% "
            f"{str(s['gaming_asymmetry_ratio']):>11} "
            f"{s['fisher_p_value']:>10.4f} "
            f"{s['cohens_h']:>7.3f} "
            f"{s['haldane_odds_ratio']:>8.3f}"
        )
    print(f"\n  Stats: {stats_path}")
    print(f"  Meta:  {meta_path}")


if __name__ == "__main__":
    main()

 
