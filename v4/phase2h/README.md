# Phase 2H — Prospective Order-Awareness Ablation

**Protocol:** `V4-P2H-CIFAR100-ORDER-ABLATION-001`

**Status:** complete; **order-awareness gate failed (5/6)**

Phase 2H tested whether directed order information adds value after symmetric composition is already modeled. The symmetric and order-aware methods used the same thirty directed pair measurements and differed by exactly one `ordered - symmetric` feature. Both forecast the complete six-coordinate behavior vector.

On 18 fresh lineages, symmetric composition reduced macro normalized MAE from 0.961 (`g0`) to 0.352. Adding order reduced it only slightly further to 0.349. Four of six coordinate MAEs, all six family mean errors, and reversal-contrast MAE improved, with aggregate improvement CI [0.00031, 0.00722]. However, macro controlled rank declined from 0.7320 to 0.7285, so the all-required gate failed. Accuracy regret was unchanged at 0.650pp.

The result supports composition-aware forecasting as the headline. Order is a small, channel-dependent secondary effect rather than an established general contribution. See `PHASE2H_FINAL_REPORT.md` and `INFORMATION_AUDIT.md`.

```bash
python3 v4/phase2h/verify_phase2h.py
python3 -m unittest discover -s v4/phase2h/tests -v
python3 -m v4.phase2h.analyze
python3 -m v4.phase2h.robustness
```
