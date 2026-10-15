#!/usr/bin/env python3
"""Single-command reproducibility runner & verification dashboard for Shift Algebra.

Usage examples:
  python reproduce.py --verify      # Instant (<1s) audit & summary of all committed results
  python reproduce.py --synthetic   # Run Synthetic Suite S1-S3
  python reproduce.py --v1          # Run original 4-primitive v1 CIFAR-10 pipeline
  python reproduce.py --v2          # Run full 6-primitive Shift Algebra v2 pipeline (Regimes II & III + 4-prim)
  python reproduce.py --v3          # Run Phase V3 Reviewer-Proofing Suite (Gaps 1-5)
  python reproduce.py --gap1        # Run individual V3 gap (--gap1, --gap2, --gap3, --gap4, --gap5)
  python reproduce.py --all         # Run S1-S3 + V1 + V2 + V3 end-to-end
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def _run_module(mod, *args):
    cmd = [sys.executable, "-m", mod, *map(str, args)]
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def verify_and_summarize():
    """Verifies all committed result artifacts, leakage audits, and prints the
    complete S1-S3, V1, V2, and V3 summary dashboard."""
    print("=" * 98)
    print("SHIFT ALGEBRA — REPRODUCIBILITY VERIFICATION & RESULTS DASHBOARD")
    print("=" * 98)

    required_files = [
        "results/summary.json",
        "results/audit_report.json",
        "results_v2/synthetic_summary.json",
        "results_v2/summary_r2.json",
        "results_v2/summary_r3.json",
        "results_v2/four_prim_summary.json",
        "results_v2/audit_report_v2.json",
        "results_v2/reviewer_suite_v3.json",
        "results_v2/REVIEWER_REPORT_V3.txt",
    ]
    missing = [p for p in required_files if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise FileNotFoundError(f"Missing expected result artifacts: {missing}")
    print(f"[check] All {len(required_files)} core result & audit JSON/TXT artifacts present.")

    # 1. Verify v1 and v2 post-hoc zero-leakage audits
    with open(os.path.join(ROOT, "results/audit_report.json")) as f:
        aud1 = json.load(f)
    with open(os.path.join(ROOT, "results_v2/audit_report_v2.json")) as f:
        aud2 = json.load(f)
    assert aud1["contamination_split_disjoint"]["pass"] is True
    assert aud1["pools_contain_only_observed"]["pass"] is True
    assert all(item.get("result") == "PASS" for item in aud2), "v2 audit_report_v2.json did not pass!"
    n_checks_v2 = len(aud2)
    print(f"[check] Zero-leakage & integrity audits: v1 data-hygiene PASS, "
          f"v2 PASS ({n_checks_v2}/{n_checks_v2} checks, 100% PASS).")

    # 2. Synthetic S1-S3 Summary
    with open(os.path.join(ROOT, "results_v2/synthetic_summary.json")) as f:
        syn = json.load(f)
    print("\n--------------------------------------------------------------------------------------------------")
    print("1. SYNTHETIC CONTROLLED SUITE (S1: Exact Semigroup, S2: Non-Commutative, S3: Saturation)")
    print("--------------------------------------------------------------------------------------------------")
    for s_key in ["S1", "S2", "S3"]:
        if s_key not in syn:
            continue
        s_val = syn[s_key]
        mb = s_val["M_B"]["bpe_mean"]
        ma = s_val["M_A"]["bpe_mean"]
        m0 = s_val["M0_meanpool"]["bpe_mean"]
        m1 = s_val["M1_behadd"]["bpe_mean"]
        sign_mb = s_val["M_B"].get("sign_match", float("nan"))
        print(f"  {s_key:<6} | M_B BPE={mb:.4f} (sign={sign_mb:.2f}) | M_A={ma:.4f} | "
              f"M0_meanpool={m0:.4f} | M1_behadd={m1:.4f}")

    # 3. 4-Primitive {A,B,C,D} Apples-to-Apples Summary
    with open(os.path.join(ROOT, "results_v2/four_prim_summary.json")) as f:
        fp = json.load(f)
    print("\n--------------------------------------------------------------------------------------------------")
    print("2. APPLES-TO-APPLES 4-PRIMITIVE {A, B, C, D} BENCHMARK (33 Held-Out Conditions, 5 Seeds)")
    print("--------------------------------------------------------------------------------------------------")
    for k, st in fp.items():
        mb = st["methods"]["M_B"]
        ma = st["methods"]["M_A"]
        m0 = st["methods"]["M0_meanpool"]
        m5 = st["methods"]["M5_v1"]
        print(f"  {st['label']}")
        print(f"    M_A BPE={ma['bpe_mean']:.4f} (ABC={ma['abc_bpe']:.4f}, sign={ma['sign_match_acc']:.2f}) | "
              f"M_B BPE={mb['bpe_mean']:.4f} (ABC={mb['abc_bpe']:.4f}, sign={mb['sign_match_acc']:.2f}, assoc={mb['assoc_err']:.4f}) | "
              f"M0={m0['bpe_mean']:.4f} | M5_v1={m5['bpe_mean']:.4f}")

    # 4. Phase V3 Reviewer-Proofing Suite (Gaps 1-5)
    with open(os.path.join(ROOT, "results_v2/REVIEWER_REPORT_V3.txt")) as f:
        v3_txt = f.read().strip()
    print("\n" + v3_txt)
    print("\n[SUCCESS] All Shift Algebra v1, v2, and v3 artifacts verified.")


def run_synthetic():
    _run_module("src.experiment_v2", "synthetic")


def run_v1():
    for phase in ["prepare-data", "train-base", "build-pools", "train-methods", "evaluate", "audit"]:
        _run_module("src.experiment", phase)
    _run_module("src.report")


def run_v2():
    _run_module("src.experiment_v2", "synthetic")
    for regime in [2, 3]:
        _run_module("src.experiment_v2", "build-pools", "--regime", regime)
        _run_module("src.experiment_v2", "train-methods", "--regime", regime)
        _run_module("src.experiment_v2", "train-transfer", "--regime", regime)
        _run_module("src.experiment_v2", "evaluate", "--regime", regime)
    _run_module("src.experiment_v2", "audit")
    _run_module("src.experiment_v2", "report")
    _run_module("src.experiment_v2", "four-prim")


def run_v3(step="all"):
    _run_module("src.reviewer_suite_v3", step)


def main():
    parser = argparse.ArgumentParser(
        description="Shift Algebra — Single-Command Reproducibility Runner"
    )
    parser.add_argument("--verify", action="store_true",
                        help="Verify all committed results, audits, and print summary dashboard (default)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Re-run Synthetic Shift Algebra Suite (S1-S3)")
    parser.add_argument("--v1", action="store_true",
                        help="Re-run original 4-primitive v1 CIFAR-10 pipeline")
    parser.add_argument("--v2", action="store_true",
                        help="Re-run full 6-primitive Shift Algebra v2 pipeline (Regimes II & III + 4-prim)")
    parser.add_argument("--v3", action="store_true",
                        help="Re-run full Phase V3 Reviewer-Proofing Suite (Gaps 1-5)")
    parser.add_argument("--gap1", action="store_true", help="Run V3 Gap 1 (14-method ablations & L_p audit)")
    parser.add_argument("--gap2", action="store_true", help="Run V3 Gap 2 (Observation-budget curve)")
    parser.add_argument("--gap3", action="store_true", help="Run V3 Gap 3 (Shift-primitive holdout test)")
    parser.add_argument("--gap4", action="store_true", help="Run V3 Gap 4 (External generalization: ViT, Fashion-MNIST, CIFAR-100)")
    parser.add_argument("--gap5", action="store_true", help="Run V3 Gap 5 (Counterfactual H & multi-level stats)")
    parser.add_argument("--all", action="store_true",
                        help="Re-run S1-S3 + V1 + V2 + V3 end-to-end")

    args = parser.parse_args()
    ran_any = False

    if args.all:
        run_v1()
        run_v2()
        run_v3("all")
        verify_and_summarize()
        return

    if args.synthetic:
        run_synthetic()
        ran_any = True
    if args.v1:
        run_v1()
        ran_any = True
    if args.v2:
        run_v2()
        ran_any = True
    if args.v3:
        run_v3("all")
        ran_any = True
    for g in ["gap1", "gap2", "gap3", "gap4", "gap5"]:
        if getattr(args, g):
            run_v3(g)
            ran_any = True
    if any(getattr(args, g) for g in ["gap1", "gap2", "gap3", "gap4", "gap5"]):
        run_v3("report")

    if args.verify or not ran_any:
        verify_and_summarize()


if __name__ == "__main__":
    main()
