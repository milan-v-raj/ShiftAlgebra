# Phase 2H Preregistration — Does Order Awareness Add Value?

**Protocol:** `V4-P2H-CIFAR100-ORDER-ABLATION-001`

## Question

When composition is already modeled, does adding directed order information improve forecasts of the complete six-coordinate behavior vector?

The experiment compares two nested, equal-calibration-budget methods:

- **symmetric composition:** model/word metadata plus symmetric pair composition, independent retention, and envelope features;
- **order-aware composition:** the identical features plus exactly one directed `ordered - symmetric` feature.

The comparison isolates incremental order information. It does not compare composition against a metadata-only baseline as its primary question.

## Development and freeze

Both methods are fit only on the already-designated Phase-2F development behavior. They share:

- metric-specific exponents;
- all fifteen unordered pair groups and all thirty directed pair measurements;
- ridge penalty and preprocessing;
- target-coordinate normalization scales;
- clean/model/recipe/word features;
- symmetric, independent, and envelope composition features.

The symmetric method averages both directions before feature construction. The order-aware method receives one additional directed-minus-symmetric channel. No Phase-2F confirmation, Phase-2G, or Phase-2H target value enters fitting. Earlier outcomes motivated this new question but are not numerical fit inputs.

Frozen method SHA-256: `9c4eef64aa2e1812306c6df3ba7c90e5f8a7bbf02f97eeece6603509d31e375e`.

## Fresh confirmation

- Dataset: canonical-provenance CIFAR-100.
- Models: 18 new lineages and 36 snapshots across the six Phase-2G architecture families and three known recipes.
- Seeds: new Phase-2H seeds, absent from prior banks.
- Images: twenty images/class at offsets 60--79, disjoint from Phases 2F and 2G.
- Targets: sixty fresh words at lengths 4--6, organized as thirty exact reversal pairs and disjoint from all prior word banks.
- Calibration: identity, six singles, and thirty directed distinct pairs.
- Metrics: margin, feature drift, ECE, balanced accuracy, accuracy, and mean confidence.

Training records and checkpoint hashes are committed before behavior evaluation. Confirmation behavior is evaluated once after that commit.

## Primary all-required gate

All six criteria must pass:

1. order-aware macro normalized MAE is below symmetric-composition MAE;
2. the architecture-family/lineage/reversal hierarchical 95% interval for symmetric-minus-order-aware normalized error excludes zero;
3. at least four of six raw coordinate MAEs improve;
4. order-aware reversal-contrast macro normalized MAE is below the symmetric method;
5. order-aware macro controlled model-rank Spearman is no worse;
6. order-aware accuracy top-1 model-selection regret is no worse.

The conclusion is binary under this protocol. If the gate fails, order is not retained as an empirically established headline contribution even if individual endpoints improve.

## Reversal-contrast endpoint

For each model, metric, and reversal group, define

`contrast = behavior(forward word) - behavior(reverse word)`.

The symmetric method predicts identical values for each reversal and therefore predicts zero contrast. The order-aware method may predict nonzero contrast. Contrast MAE is normalized by the Phase-2F development coordinate scales and macro-averaged over all six metrics. A hierarchical contrast-improvement interval is reported descriptively.

## Inference

Ten thousand deterministic bootstrap draws resample architecture family, lineage within sampled family, and reversal group as a crossed target axis. Epoch snapshots stay inside lineages. Per-coordinate raw MAE, aggregate normalized MAE, controlled model rank, accuracy regret, family signs, reversal-contrast errors, and unique-coordinate sensitivity are retained regardless of outcome.

## Interpretation boundary

Passing would show that directed order information adds value beyond an already composition-aware symmetric model in this CIFAR-100 synthetic-shift regime. Failing would support simplifying the paper to composition-aware, order-agnostic forecasting. Neither outcome establishes natural-shift, cross-dataset, ImageNet, or universal-algebra validity.
