# Phase 2H — Prospective Order-Awareness Ablation

**Protocol:** `V4-P2H-CIFAR100-ORDER-ABLATION-001`

**Status:** methods and protocol frozen; confirmation behavior unopened

Phase 2H asks whether directed order information adds value after symmetric composition is already modeled. The symmetric and order-aware methods have the same calibration conditions and differ by exactly one `ordered - symmetric` feature. Both predict the complete six-coordinate behavior vector.

The frozen comparison uses Phase-2F development outcomes only. Confirmation uses 18 fresh lineages, CIFAR-100 image offsets 60--79, and sixty fresh words arranged into thirty reversal pairs.

```bash
python3 v4/phase2h/verify_phase2h.py
python3 -m unittest discover -s v4/phase2h/tests -v
```
