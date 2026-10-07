"""Readable summary of experiment results.

Usage: python -m src.report
Reads results/metrics_summary.csv, results/stats_vs_m5.csv,
results/summary.json, results/behavior_testpool.csv and prints a formatted
report (also saved to results/REPORT.txt).
"""
import csv
import json
import os

from . import config as cfg


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    summary = json.load(open(os.path.join(cfg.RESULTS_DIR, "summary.json")))
    out("=" * 78)
    out("SHIFT ALGEBRA — EXPERIMENT RESULTS (Benchmark V, CIFAR-10)")
    out("=" * 78)
    out(f"base model: ResNet-18 width={cfg.BASE_MODEL['width']} "
        f"(Amendment 1), clean test acc = {summary['clean_test_acc']*100:.2f}%  "
        f"ECE = {summary['clean_test_ece']:.4f}  conf = {summary['clean_test_conf']:.4f}")
    out(f"split: 45 observed conditions (4 prim x5 sev + 5 pairs x5 sev), "
        f"33 held-out (BC/CB x5, 4 triples x5, 3 off-grid)")
    out("")

    out("-" * 78)
    out("METHOD SUMMARY (mean over 5 seeds)")
    out("-" * 78)
    hdr = (f"{'method':<20} {'BPE':>10} {'accMAE':>8} {'rankρ':>7} "
           f"{'CGG':>10} {'RCE':>7} {'OSIabs':>8} {'IREpr':>9} {'IREtr':>9}")
    out(hdr)
    for r in summary["methods"]:
        rho = "  n/a " if r["rank_rho"] is None else f"{r['rank_rho']:7.3f}"
        cr = "  n/a " if r["commutator_rho"] is None else f"{r['commutator_rho']:7.3f}"
        out(f"{r['method']:<20} {r['bpe_mean']:>10.6f} {r['acc_mae_mean']:>8.4f} "
            f"{rho} {r['cgg_mean']:>10.6f} {r['rce_mean']:>7.3f} "
            f"{r['osi_abs']:>8.4f} {r['ire_pair']:>9.6f} {r['ire_triple']:>9.6f}")
    out(f"{'oracle (not zero-shot)':<20} "
        f"{summary['methods'][0]['oracle_bpe_mean']:>10.6f}")
    out("")

    out("-" * 78)
    out("STATISTICAL COMPARISONS vs M5 (paired Wilcoxon over 33 held-out conditions,")
    out("seed-averaged per-condition squared error; 95% bootstrap CI of mean diff;")
    out("diff = M5_err - other_err, so POSITIVE = M5 WORSE, NEGATIVE = M5 BETTER)")
    out("-" * 78)
    for r in summary["stats"]:
        out(f"{r['comparison']:<32} diff={r['mean_diff']:+.6f}  "
            f"CI=[{r['ci_lo']:+.6f},{r['ci_hi']:+.6f}]  p={r['p_value']:.4g}")
    out("")

    out("-" * 78)
    out("DECISION RULES (pre-registered, plan §7)")
    out("-" * 78)
    for k, v in summary["decision_rules"].items():
        out(f"  {'PASS' if v else 'FAIL'}  {k}")
    out("")

    out("-" * 78)
    out("BEHAVIOR TABLE (measured on full 10k test pool; selected rows)")
    out("-" * 78)
    with open(os.path.join(cfg.RESULTS_DIR, "behavior_testpool.csv")) as f:
        rows = list(csv.DictReader(f))
    out(f"{'condition':<16} {'kind':<8} {'acc':>7} {'d_acc':>8} {'ece':>7} "
        f"{'drift':>7} {'conf':>7}")
    show = [r for r in rows if r["letters"] in
            ("A", "B", "C", "D", "B+C", "C+B", "A+B+C", "A+B+D", "A+C+D", "B+C+D")]
    # one row per condition at default severity (sev index 2), plus all orders
    for r in show:
        sev2 = [x for x in rows if x["letters"] == r["letters"]]
        mid = sev2[len(sev2) // 2]
        out(f"{mid['letters']:<16} {mid['kind']:<8} {float(mid['acc'])*100:>6.2f}% "
            f"{float(mid['d_acc'])*100:>7.2f}pp {float(mid['ece']):>7.4f} "
            f"{float(mid['feature_drift']):>7.3f} {float(mid['mean_conf']):>7.4f}")

    txt = "\n".join(lines)
    with open(os.path.join(cfg.RESULTS_DIR, "REPORT.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\n[saved] {os.path.join(cfg.RESULTS_DIR, 'REPORT.txt')}")


if __name__ == "__main__":
    main()
