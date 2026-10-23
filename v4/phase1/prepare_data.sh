#!/usr/bin/env bash
set -euo pipefail

# Transport-only mirrors used because the execution environment permits GitHub
# but not the canonical dataset hosts. Dataset files remain ignored under data/.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL="$ROOT/data/external"
mkdir -p "$EXTERNAL"

CIFAR_COMMIT="fad27e50dfca620330fa5f824432abcf1781c161"
EMNIST_COMMIT="0f121efc642634809af4f2f443b70152f71b9a51"

if [[ ! -d "$EXTERNAL/cifar-mirror/.git" ]]; then
  git clone --depth 1 https://github.com/wikiabhi/Cifar-10.git "$EXTERNAL/cifar-mirror"
fi
git -C "$EXTERNAL/cifar-mirror" fetch --depth 1 origin "$CIFAR_COMMIT"
git -C "$EXTERNAL/cifar-mirror" checkout --detach "$CIFAR_COMMIT"

if [[ ! -d "$EXTERNAL/emnist-packed/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/0r4nd/emnist-packed.git "$EXTERNAL/emnist-packed"
fi
git -C "$EXTERNAL/emnist-packed" sparse-checkout set emnist-balanced
git -C "$EXTERNAL/emnist-packed" fetch --depth 1 origin "$EMNIST_COMMIT"
git -C "$EXTERNAL/emnist-packed" checkout --detach "$EMNIST_COMMIT"

printf 'CIFAR transport commit: %s\n' "$(git -C "$EXTERNAL/cifar-mirror" rev-parse HEAD)"
printf 'EMNIST transport commit: %s\n' "$(git -C "$EXTERNAL/emnist-packed" rev-parse HEAD)"
printf 'Dataset loaders will verify file hashes/counts before use.\n'
