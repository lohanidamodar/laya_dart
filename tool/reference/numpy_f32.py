"""numpy's float32 `exp` and `log`, which laya's ONNX decode runs on the head's float32 output.

numpy's float32 kernels are not correctly rounded (they are SIMD polynomial kernels, up to a few
ulp off), so the Dart port reproduces them bit for bit. This records numpy's own outputs as uint32
bit patterns: seeded random inputs over the ranges the decode sees (`exp` of a max-subtracted row,
`log` of a probability clipped to [1e-12, 1]), plus inputs that once exposed a difference.

The kernels are CPU-dependent: these were recorded with numpy's AVX2/FMA3 kernels (always the
path on Windows builds, where the AVX512F kernels are compiled out).
"""
import numpy as np

from _shared import emit

# exp(-72.43388f) is where numpy's compiler-fused `x * log2e + magic` differs from rounding the
# product first.
EXP_EDGES = [-72.43388366699219, 0.0, -0.0, -1e-8, -87.3, -88.0, -100.0, -103.97208404541015625,
             -103.9, -104.0, -1e4, 88.0, 88.72283935546875, 1.0, -1.0, 0.5, -0.5]
LOG_EDGES = [1.0, 0.5, 0.7071067690849304, 0.7071067094802856, 1e-12, 1e-30, 1e-40, 2.0,
             0.25, 0.9999999403953552, 1.0000001192092896]


def main():
    rng = np.random.default_rng(20261003)
    exp_in = np.concatenate([np.array(EXP_EDGES), rng.uniform(-110, 0, 1500),
                             rng.uniform(-1, 1, 500), -np.abs(rng.normal(0, 5, 500))])
    log_in = np.concatenate([np.array(LOG_EDGES), rng.uniform(0, 1, 1000),
                             10 ** rng.uniform(-12, 0, 1000), rng.uniform(0.5, 1.5, 500)])
    cases = []
    for fn, xs in (("exp", exp_in), ("log", log_in)):
        x = np.ascontiguousarray(xs.astype(np.float32))
        with np.errstate(all="ignore"):
            y = getattr(np, fn)(x)
        cases.append({"fn": fn, "in": x.view(np.uint32).tolist(), "out": y.view(np.uint32).tolist()})
    emit(cases)


if __name__ == "__main__":
    main()
