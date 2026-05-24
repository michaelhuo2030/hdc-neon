"""Benchmark hdc_neon vs numpy LUT256 reference.

Reproduces the headline result: 89× speedup at N=200K, D=10K on M-series Mac.

Usage:
    python examples/benchmark.py                # full N=200K
    python examples/benchmark.py --quick        # N=50K for faster iteration
"""
from __future__ import annotations

import argparse
import time
import numpy as np

try:
    import hdc_neon
except ImportError:
    print("ERROR: hdc_neon not installed.")
    print("Install with: pip install hdc-neon")
    print("Or build from source: maturin develop --release")
    raise SystemExit(1)


# Precomputed 256-entry popcount LUT for numpy reference
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def numpy_lut256_search(docs_packed_u8, q_packed_u8, k=5):
    """numpy bit-packed XOR + LUT256 popcount baseline."""
    def search(qi):
        xor = np.bitwise_xor(docs_packed_u8, q_packed_u8[qi])
        hamming = _POPCOUNT_LUT[xor].sum(axis=1, dtype=np.uint16)
        return np.argpartition(hamming, k)[:k].tolist()
    return search


def neon_search(docs_u64, q_u64, k=5):
    """Rust hdc_neon hamming_topk."""
    def search(qi):
        return hdc_neon.hamming_topk(docs_u64, q_u64[qi], k).tolist()
    return search


def bench(name, search_fn, n_queries, runs=3):
    # warmup
    search_fn(0)
    times = []
    for run in range(runs):
        t0 = time.perf_counter()
        for qi in range(n_queries):
            search_fn(qi)
        times.append(time.perf_counter() - t0)
    ms = float(np.median(times) / n_queries * 1000)
    qps = 1000 / ms
    print(f"  {name:<24}  {ms:8.3f} ms/q   {qps:7.1f} qps")
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-docs", type=int, default=200_000)
    p.add_argument("--d-hdc", type=int, default=10_240)
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    N = 50_000 if args.quick else args.n_docs
    D = ((args.d_hdc + 63) // 64) * 64  # round to multiple of 64
    Q = args.n_queries

    print(f"\n=== hdc_neon benchmark ===")
    print(f"N={N:,} docs, D={D:,} bits, Q={Q} queries\n")

    # Generate synthetic data
    rng = np.random.default_rng(7)
    d_in = 384
    docs = rng.standard_normal((N, d_in)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    queries = rng.standard_normal((Q, d_in)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Random projection → bipolar HVs
    R = np.random.default_rng(42).standard_normal((d_in, D)).astype(np.float32)
    doc_bits = (docs @ R > 0).astype(np.uint8)
    q_bits = (queries @ R > 0).astype(np.uint8)

    # Bit-packed forms
    docs_u8 = np.packbits(doc_bits, axis=1)
    q_u8 = np.packbits(q_bits, axis=1)
    docs_u64 = np.ascontiguousarray(docs_u8).view(np.uint64).copy()
    q_u64 = np.ascontiguousarray(q_u8).view(np.uint64).copy()

    print(f"docs (bit-packed): {docs_u64.shape} = {docs_u64.nbytes/1e6:.1f} MB")
    print()

    numpy_ms = bench("numpy LUT256 (baseline)", numpy_lut256_search(docs_u8, q_u8), Q)
    neon_ms = bench("hdc_neon NEON",           neon_search(docs_u64, q_u64), Q)

    speedup = numpy_ms / neon_ms
    print(f"\nSpeedup: {speedup:.1f}× over numpy")
    if speedup > 50:
        print("✅ Confirmed: hdc_neon is significantly faster than numpy.")
    elif speedup > 5:
        print("⚠️  Lower speedup than expected (target 50-100×). Check Rust release build.")
    else:
        print("❌ Unexpected: speedup too low. Did you build with --release?")


if __name__ == "__main__":
    main()
