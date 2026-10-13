"""V2 primitive corruptions + composition application (PLAN_V2 §5.1).

Extends v1 (src/corruptions.py) with two primitives:
  P  pixelate   block downsample + nearest upsample (severity = block size)
  S  saturation interpolate away from per-pixel grayscale (severity = factor)

Conventions are unchanged from v1 (Appendix A.2):
- photometric ops act on float images in [0, 1]; every op clips to [0, 1];
- compositions apply primitives in the listed order, clipping each step;
- noise (C) uses the order-independent condition seed (frozenset + sorted
  (letter, severity) pairs), so X∘C and C∘X share the identical noise field
  and their difference is a pure order effect (verified by the audit phase).
"""
import zlib
import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(sigma: float) -> torch.Tensor:
    radius = max(1, int(3.0 * sigma + 0.5))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float64)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).float()


def apply_primitive(x: torch.Tensor, letter: str, sev: float,
                    generator: torch.Generator = None) -> torch.Tensor:
    """x: (N, 3, 32, 32) float in [0,1] -> corrupted tensor in [0,1]."""
    if letter == "A":                      # brightness: photometric scaling
        return (x * sev).clamp(0.0, 1.0)
    if letter == "D":                      # contrast about per-image mean
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        return (mu + sev * (x - mu)).clamp(0.0, 1.0)
    if letter == "B":                      # gaussian blur, separable, reflect pad
        k = _gaussian_kernel_1d(sev)
        r = (k.numel() - 1) // 2
        w_h = k.view(1, 1, 1, -1).expand(3, 1, 1, -1).contiguous()
        w_v = k.view(1, 1, -1, 1).expand(3, 1, -1, 1).contiguous()
        x = F.pad(x, (r, r, r, r), mode="reflect")
        x = F.conv2d(x, w_h, padding=0, groups=3)
        x = F.conv2d(x, w_v, padding=0, groups=3)
        return x.clamp(0.0, 1.0)
    if letter == "C":                      # additive gaussian noise
        if generator is None:
            raise ValueError("noise corruption requires a deterministic generator")
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype) * sev
        return (x + noise).clamp(0.0, 1.0)
    if letter == "P":                      # pixelate: fractional down + nearest up
        if sev <= 1.0:
            return x
        h, w = x.shape[-2], x.shape[-1]
        small = max(2, int(round(h / float(sev))))
        xd = F.interpolate(x, size=(small, small), mode="bilinear", align_corners=False)
        return F.interpolate(xd, size=(h, w), mode="nearest").clamp(0.0, 1.0)
    if letter == "S":                      # saturation about per-pixel gray
        gray = x.mean(dim=1, keepdim=True)
        return (gray + sev * (x - gray)).clamp(0.0, 1.0)
    raise ValueError(f"unknown primitive {letter!r}")


def condition_seed(letters, sevs) -> int:
    """Order-independent deterministic seed for a condition's noise field."""
    fset_str = "frozenset({" + ", ".join(repr(x) for x in sorted(set(letters))) + "})"
    pairs_str = repr(tuple(sorted(zip(letters, (float(s) for s in sevs)))))
    key = f"({fset_str}, {pairs_str})"
    return zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF


def apply_condition(x: torch.Tensor, letters, sevs) -> torch.Tensor:
    """Apply a composition: primitives in the listed order, clipping each step."""
    assert len(letters) == len(sevs), "letters/sevs length mismatch"
    out = x
    if "C" in letters:
        g = torch.Generator().manual_seed(condition_seed(letters, sevs))
    else:
        g = None
    for letter, sev in zip(letters, sevs):
        out = apply_primitive(out, letter, sev, generator=g if letter == "C" else None)
    return out
