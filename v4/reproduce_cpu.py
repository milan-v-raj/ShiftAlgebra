#!/usr/bin/env python3
"""Verify, reanalyze, and summarize the committed V4 CPU benchmark."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def command(*args):
 print(">>>"," ".join(map(str,args)),flush=True); subprocess.run([str(x) for x in args],cwd=ROOT,check=True)

def verify():
 for script in ("v4/phase2/verify_phase2.py","v4/phase2b/verify_phase2b.py","v4/phase2c/verify_phase2c.py","v4/phase2d/verify_phase2d.py","v4/verify_cpu_synthesis.py"):
  command(sys.executable,script)
 command(sys.executable,"-m","unittest","discover","-s","v4/phase2b/tests","-q")
 command(sys.executable,"-m","unittest","discover","-s","v4/phase2c/tests","-q")
 command(sys.executable,"-m","unittest","discover","-s","v4/phase2d/tests","-q")
 print("V4 CPU verification complete")

def reanalyze():
 for module in ("v4.phase2b.analyze","v4.phase2c.analyze","v4.phase2d.analyze","v4.cpu_synthesis"):
  command(sys.executable,"-m",module)

def summary():
 syn=json.loads((ROOT/"v4/results/CPU_CROSS_PHASE_SYNTHESIS.json").read_text())
 print("="*72); print("V4 CPU COMPOSITIONAL FORECASTING SUMMARY"); print("="*72)
 for bank,row in syn["banks"].items():
  print(f"{bank:8s} lineages={row['lineages']:2d} words={row['words']:2d} g0={row['g0']['accuracy_mae_pp']:.3f}pp g1={row['g1']['accuracy_mae_pp']:.3f}pp")
 macro=syn["macro_bank_mae_pp"]; ci=syn["incremental_meta_bootstrap"]["ci95_pp"]
 print(f"macro-bank MAE: {macro['g0']:.3f}pp -> {macro['g1']:.3f}pp")
 print(f"retrospective bootstrap sensitivity interval: [{ci[0]:.3f}, {ci[1]:.3f}]pp")
 print("warning: four banks share CIFAR-10; this is not independent-dataset confirmation")
 print("gates:",syn["gate_summary"])

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--verify",action="store_true"); p.add_argument("--reanalyze",action="store_true"); p.add_argument("--summary",action="store_true"); p.add_argument("--all",action="store_true"); a=p.parse_args()
 if not any(vars(a).values()): a.verify=a.summary=True
 if a.all or a.verify: verify()
 if a.all or a.reanalyze: reanalyze()
 if a.all or a.summary: summary()
if __name__=="__main__": main()
