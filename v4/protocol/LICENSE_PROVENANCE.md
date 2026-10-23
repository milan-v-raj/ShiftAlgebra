# Phase 1 License and Provenance Audit

**Audit date:** 2026-10-23
**Protocol:** `V4-P1-MASON-001`
**Status:** open; this document prevents an unjustified “exact reproduction” claim

## Upstream repository

- Repository: <https://github.com/ianxmason/compositional-robustness>
- Pinned commit: `fc62a92ccd11f64eec26c7a019e9e0957fd9bd33`
- Relevant word generator: `src/data/sample_corruptions.py`
- Relevant corruption definitions: `src/data/data_transforms.py`
- Environment: `pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime` plus packages in `docker/Dockerfile`
- Upstream random seed for words: `2778`

### License finding

GitHub's license endpoint returns HTTP 404 and the pinned tree contains no detected license file. Therefore:

1. do not vendor or redistribute upstream source code;
2. treat inspected code as protocol documentation;
3. implement clean adapters independently;
4. cite the paper/repository;
5. seek author permission before redistributing derived upstream code or assets.

The generated word manifest records factual tuples and provenance; it does not include the upstream implementation.

## Checkpoints

The upstream README describes commands that train and save checkpoints but does not publish checkpoint download URLs. The repository tree contains no checkpoint files.

**Decision:** Phase 1 must train independent checkpoints from declared implementations, or obtain author-provided checkpoints with hashes. Results cannot be called a bitwise checkpoint reproduction unless official weights are obtained.

## Corruption defaults

Pinned constructor defaults:

| Primitive | Default severity/setting |
|---|---:|
| Contrast | 2 |
| GaussianBlur | 2 |
| ImpulseNoise | 5 |
| Invert | fixed |
| Rotate90 | fixed 90 degrees |
| Swirl | 3 |

The manifest stores these values explicitly. Reverse-order stochastic conditions must share a deterministic per-image noise seed in the new benchmark, even if the historical data-generation process did not expose one.

## Dataset status

### EMNIST

- Upstream loader uses the torchvision EMNIST distribution.
- Download/version/checksum and underlying NIST terms remain to be pinned before inference.
- Data must not be committed to Git.

**Status:** usable after checksum/terms verification.

### CIFAR-10

- Upstream loader uses torchvision `CIFAR10`.
- The upstream split takes the first 4,000 sorted training examples per class for training and the last 1,000 per class for validation; the standard test set remains test.
- Download archive checksum and dataset terms remain to be recorded.
- Data must not be committed to Git.

**Status:** usable after checksum/terms verification.

### FaceScrub

The upstream README says:

- FaceScrub cannot be downloaded directly and access must be requested;
- the authors used an internal subset;
- identities with fewer than 100 images were removed;
- the resulting task has 388 identities numbered 100--487.

That exact internal subset is not available in this repository and cannot be reconstructed reliably from the description alone.

**Status:** blocked. Do not claim exact FaceScrub reproduction. Run EMNIST/CIFAR first; use FaceScrub only after lawful access and an exact sample manifest are obtained. A replacement face dataset would be a new external test, not a reproduction.

## Training protocol facts found upstream

The monolithic training script declares 200 epochs and:

| Dataset | Classes | CE learning rate | Contrastive learning rate | Contrastive weight |
|---|---:|---:|---:|---:|
| EMNIST | 47 | `1e-2` | `1e-2` | `1e-1` |
| CIFAR-10 | 10 | `1e-1` | `1e-1` | `1e-1` |
| FaceScrub | 388 | `1e-1` | `1e-1` | `1e-2` |

These shell settings are not a complete training contract. Optimizer, scheduler, augmentation, network construction, and all random seeds must be extracted and recorded before freeze.

## Required closure evidence

- [ ] author/repository license clarified;
- [ ] EMNIST archive/version/checksum and terms recorded;
- [ ] CIFAR-10 archive/version/checksum and terms recorded;
- [ ] FaceScrub access decision recorded;
- [ ] model implementation provenance recorded;
- [ ] optimizer/scheduler/augmentation settings recorded;
- [ ] checkpoint hashes recorded;
- [ ] preprocessing and class mappings recorded;
- [ ] stochastic corruption random-field policy frozen.
