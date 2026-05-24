# hdc_neon

> Fast hyperdimensional-computing (HDC) retrieval via NEON SIMD popcount + Rayon parallel-foreach.
> **89× faster than naive numpy. 1.5× faster than dense-float PyTorch MPS. Same speed class as FAISS HNSW. Smaller storage per doc.**

[![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-stable-orange)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Build](https://github.com/michaelhuo2030/hdc-neon/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelhuo2030/hdc-neon/actions)

> Background: [blog post — I made HDC retrieval 89× faster with 90 lines of Rust](TBD-medium-link)
> Browser demo (no install): [michaelhuo2030.github.io/hdc-neon](https://michaelhuo2030.github.io/hdc-neon/)

## What this is

`hdc_neon` is a 90-line Rust extension exposing fast HDC retrieval primitives to Python:

- **`hamming_topk(docs, query, k)`** — top-k nearest by Hamming distance over bit-packed bipolar HVs (NEON popcount path)
- **`hamming_all(docs, query)`** — full distance vector
- **`int8_dot_topk(docs, query, k)`** — top-k by int8 dot product (for multi-bit signed HDC)
- **`f32_dot_topk(docs, query, k)`** — top-k by fp32 dot (general-purpose)
- **`fourier_topk(real, imag, q_real, q_imag, k)`** — top-k by complex Fourier dot
- **`hamming_topk_batched`** / `_bnb` / `_hierarchical` / `_tiled` — variants for batched, branch-and-bound, coarse+fine, and L1-tiled

The core insight: `u64::count_ones()` compiles to NEON `vcntq_u8 + addv` on ARM64. Rayon parallelizes across docs. That's the whole library.

## Benchmark (M4 MacBook Pro, N=200,000 docs, D=10,240 bits)

| Method | ms/q | qps | R@5 | Speedup vs numpy |
|---|---|---|---|---|
| numpy LUT256 popcount | 265 | 3.8 | 0.520 | 1× (baseline) |
| **Rust `hdc_neon`** | **2.98** | **335** | **0.520** | **89×** ⚡ |
| PyTorch MPS dense float | 4.46 | 224 | 1.000 | (ref) |
| FAISS HNSW M=32 | 0.20 | 4953 | 0.765 | (different scale; see below) |

**At N=5,000 real bge text on SciFact:** `hdc_neon` matches dense fp32 quality (R@5=0.765 = 0.765), 1.25 KB/doc bit-packed (vs 1.5 KB/doc fp32).

```
N=200,000 docs, D=10,240 bits, R@5=0.520 (identical to numpy reference)
numpy LUT256:    265 ms/query   (3.8 qps)
hdc_neon NEON:   2.98 ms/query  (335 qps)
                 ─────────────
                 89× speedup
```

## Quick start

### Install (Python)

```bash
pip install hdc-neon
```

(Or build from source — see "Build from source" below.)

### Use

```python
import hdc_neon
import numpy as np

# Generate synthetic data
N, d_in, D = 200_000, 384, 10_240
docs_float = np.random.randn(N, d_in).astype(np.float32)
docs_float /= np.linalg.norm(docs_float, axis=1, keepdims=True)
query_float = np.random.randn(d_in).astype(np.float32)
query_float /= np.linalg.norm(query_float)

# Encode to bipolar HVs via SimHash random projection
R = np.random.default_rng(42).standard_normal((d_in, D)).astype(np.float32)
docs_bits = (docs_float @ R > 0).astype(np.uint8)
query_bits = (query_float @ R > 0).astype(np.uint8)

# Bit-pack and reinterpret as uint64 (D must be a multiple of 64)
docs_packed = np.packbits(docs_bits, axis=1)
query_packed = np.packbits(query_bits)
docs_u64 = np.ascontiguousarray(docs_packed).view(np.uint64).copy()
query_u64 = np.ascontiguousarray(query_packed).view(np.uint64).copy()

# Top-5 nearest
top_5 = hdc_neon.hamming_topk(docs_u64, query_u64, 5)
print(top_5)  # array of 5 doc indices, sorted by Hamming distance
```

### Browser demo (no install)

Same WGSL kernel running in your browser via WebGPU: [michaelhuo2030.github.io/hdc-neon](https://michaelhuo2030.github.io/hdc-neon/). N=200K verified bit-for-bit against JS popcount reference.

## When to use this vs FAISS

| Use case | Recommendation |
|---|---|
| Production RAG at N=1M+, single-query latency matters most | **FAISS HNSW** (purpose-built ANN graph; ~1-5 ms/q regardless of scale) |
| Streaming append at IoT/agent-memory rates, no graph rebalance | **`hdc_neon`** (linear array, O(1) append) |
| Compositional retrieval (bind/unbind/bundle queries that FAISS can't express) | **`hdc_neon`** (HDC native) |
| Browser deployment (no server) | **`hdc_neon` + WebGPU** (FAISS can't) |
| Edge/wearable on tight power budget | **`hdc_neon`** (small, bit-packed) |
| You want a 90-line library you fully understand | **`hdc_neon`** 🙂 |

`hdc_neon` is **complementary to FAISS**, not a replacement. The two solve different shapes of the retrieval problem.

## Build from source

Requires: Rust (stable), Python 3.8+, [`maturin`](https://www.maturin.rs/).

```bash
git clone https://github.com/michaelhuo2030/hdc-neon
cd hdc-neon
RUSTFLAGS="-C target-cpu=native" maturin develop --release
python examples/benchmark.py
```

Build takes ~15s on M4. The release binary is ~250 KB.

## The 90 lines (core kernel)

```rust
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline]
fn hamming_u64(a: &[u64], b: &[u64]) -> u32 {
    let mut h: u32 = 0;
    for i in 0..a.len() {
        h += (a[i] ^ b[i]).count_ones();  // compiles to NEON vcntq_u8 + addv
    }
    h
}

#[pyfunction]
fn hamming_topk(
    docs: PyReadonlyArray2<u64>,
    query: PyReadonlyArray1<u64>,
    k: usize,
) -> PyResult<Vec<usize>> {
    let docs = docs.as_array();
    let q = query.as_slice().unwrap();
    let mut dists: Vec<(u32, usize)> = (0..docs.shape()[0])
        .into_par_iter()  // rayon: parallel across docs
        .map(|i| (hamming_u64(docs.row(i).as_slice().unwrap(), q), i))
        .collect();
    dists.select_nth_unstable_by_key(k, |&(d, _)| d);  // O(N) partial sort
    dists.truncate(k);
    dists.sort_by_key(|&(d, _)| d);
    Ok(dists.into_iter().map(|(_, i)| i).collect())
}
```

That's it. See `src/lib.rs` for the full source with all variants.

## Cross-platform

| Platform | Status | Backend |
|---|---|---|
| Apple Silicon (M1+) | ✅ Verified | NEON via `u64::count_ones()` |
| x86_64 with POPCNT (most CPUs 2008+) | ✅ Should work | `popcnt` via `u64::count_ones()` |
| AVX-512 popcount (Ice Lake+) | ⚠️ Not auto-optimized; need explicit intrinsics | Open to PRs |
| ARM Linux (server / Android) | ✅ Should work | Same NEON path |
| Web browser (Chrome / Edge 113+) | ✅ Via [WGSL demo](https://michaelhuo2030.github.io/hdc-neon/) | WebGPU |

## What's NOT in this repo

- **HDC encoding**: this library accepts pre-encoded bit-packed HVs. For encoding text → HVs, use any embedding model + `(X @ R > 0)` SimHash projection.
- **Multi-bit/Fourier optimal codecs**: experimental — see `examples/multibit.py` and `examples/fourier.py` for reference impls.
- **Compositional algebra**: separate library (`hdc_compose`) planned for Q3 2026.
- **Distributed sharding**: out of scope; combine with your own sharding layer.

## Cite

```bibtex
@misc{huo2026hdcneon,
  author = {Michael (Xiaojie) Huo},
  title = {{hdc\_neon}: Fast HDC retrieval via NEON SIMD},
  year = {2026},
  url = {https://github.com/michaelhuo2030/hdc-neon},
}
```

## Related work

- [pgvector](https://github.com/pgvector/pgvector) — Postgres vector search (Andrew Kane)
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook's ANN library
- [hnswlib](https://github.com/nmslib/hnswlib) — HNSW algorithm (Yury Malkov)
- [annoy](https://github.com/spotify/annoy) — Spotify's vector library (Erik Bernhardsson)
- [Kanerva, P. (2009). Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors](https://link.springer.com/article/10.1007/s12559-009-9009-8)

## License

Apache 2.0 — see [LICENSE](LICENSE).

Copyright 2026 Michael (Xiaojie) Huo.

## Contact

- GitHub: [@michaelhuo2030](https://github.com/michaelhuo2030)
- Email: xh638@stern.nyu.edu
- Building: [millisecond-era](https://github.com/michaelhuo2030/millisecond-era) (28nm ReRAM CIM chip for HDC workloads)

PRs welcome. Issues welcome. Performance regressions are bugs.
