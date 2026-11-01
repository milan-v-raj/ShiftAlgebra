#!/usr/bin/env python3
"""Verify, reanalyze, and summarize the committed V4 CPU benchmark."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def command(*args):
 print(">>>"," ".join(map(str,args)),flush=True); subprocess.run([str(x) for x in args],cwd=ROOT,check=True)

def verify():
 for script in ("v4/phase2/verify_phase2.py","v4/phase2b/verify_phase2b.py","v4/phase2c/verify_phase2c.py","v4/phase2d/verify_phase2d.py","v4/phase2e/verify_phase2e.py","v4/phase2f/verify_phase2f.py","v4/verify_cpu_synthesis.py"):
  command(sys.executable,script)
 command(sys.executable,"-m","unittest","discover","-s","v4/phase2b/tests","-q")
 command(sys.executable,"-m","unittest","discover","-s","v4/phase2c/tests","-q")
 command(sys.executable,"-m","unittest","discover","-s","v4/phase2d/tests","-q")
 command(sys.executable,"-m","unittest","discover","-s","v4/phase2e/tests","-q")
 command(sys.executable,"-m","unittest","discover","-s","v4/phase2f/tests","-q")
 print("V4 CPU verification complete")

def reanalyze():
 for module in ("v4.phase2b.analyze","v4.phase2c.analyze","v4.phase2d.analyze","v4.phase2e.analyze","v4.phase2f.analyze","v4.phase2f.robustness","v4.architecture_synthesis","v4.cpu_synthesis"):
  command(sys.executable,"-m",module)

def summary():
 syn=json.loads((ROOT/"v4/results/CPU_CROSS_PHASE_SYNTHESIS.json").read_text()); arch=json.loads((ROOT/"v4/results/ARCHITECTURE_SYNTHESIS.json").read_text())
 print("="*72); print("V4 CPU COMPOSITIONAL FORECASTING SUMMARY"); print("="*72)
 for bank,row in syn["banks"].items():
  print(f"{bank:8s} lineages={row['lineages']:2d} words={row['words']:2d} g0={row['g0']['accuracy_mae_pp']:.3f}pp g1={row['g1']['accuracy_mae_pp']:.3f}pp")
 macro=syn["macro_bank_mae_pp"]; ci=syn["incremental_meta_bootstrap"]["ci95_pp"]
 print(f"macro-bank MAE: {macro['g0']:.3f}pp -> {macro['g1']:.3f}pp")
 print(f"retrospective bootstrap sensitivity interval: [{ci[0]:.3f}, {ci[1]:.3f}]pp")
 print("warning: five banks share CIFAR-10; this is not independent-dataset confirmation")
 print(f"architecture sensitivity: {arch['families_favoring_g1']}/{arch['families']} families favor g1; sign p={arch['exact_one_sided_sign_p_assuming_independent_families']:.4f}")
 p2f=json.loads((ROOT/"v4/phase2f/results/PHASE2F_ANALYSIS.json").read_text())
 print(f"CIFAR-100 full vector: g0={p2f['methods']['g0']['macro_normalized_mae']:.3f} -> composition={p2f['methods']['full15']['macro_normalized_mae']:.3f}; gate={p2f['status']}")
 print("CIFAR-100 magnitude selector gate:","PASS" if p2f['magnitude_confirmation']['gates']['magnitude_gate_passed'] else "FAIL")
 print("CIFAR-10 gates:",syn["gate_summary"])

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--verify",action="store_true"); p.add_argument("--reanalyze",action="store_true"); p.add_argument("--summary",action="store_true"); p.add_argument("--all",action="store_true"); a=p.parse_args()
 if not any(vars(a).values()): a.verify=a.summary=True
 if a.all or a.verify: verify()
 if a.all or a.reanalyze: reanalyze()
 if a.all or a.summary: summary()
if __name__=="__main__": main()
