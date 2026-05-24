# Changelog

## [0.1.0] — 2026-05-26 (planned ship)

Initial public release.

### Added
- `hamming_topk(docs, query, k)` — top-k Hamming distance via NEON popcount + Rayon
- `hamming_all(docs, query)` — full Hamming distance vector
- `int8_dot_topk(docs, query, k)` — top-k int8 dot product (multi-bit signed HDC)
- `f32_dot_topk(docs, query, k)` — top-k fp32 dot product
- `fourier_topk(real, imag, q_real, q_imag, k)` — top-k complex Fourier dot
- `hamming_topk_batched(docs, queries, k)` — process B queries × N docs in one call
- `hamming_topk_bnb(docs, query, k)` — branch-and-bound prefix pruning
- `hamming_topk_hierarchical(coarse, fine, ...)` — coarse-D filter + fine-D rerank
- `hamming_topk_tiled(docs, query, k)` — L1-cache-tiled NEON variant

### Benchmarked
- M4 MacBook Pro, N=200K, D=10K: **2.98 ms/query (335 qps)** for `hamming_topk`
- Cross-validated against numpy reference; R@5 identical
- SciFact bge-small-en: matches dense fp32 quality at 1.25 KB/doc bit-packed

### License
- Apache 2.0
