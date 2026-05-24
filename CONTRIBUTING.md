# Contributing

Thanks for considering a contribution to `hdc_neon`.

## What's welcome

- **Bug reports** — file an issue with a minimal repro
- **Benchmark replications** on your hardware (especially non-M-series ARM, x86, NVIDIA GPU) — open a PR adding your result to `benchmarks/`
- **Performance improvements** — PRs that show ≥5% speedup with a benchmark diff
- **API additions** — small, focused PRs (one new function per PR)
- **Documentation** — typo fixes, clarifications, more examples
- **Cross-platform tests** — Linux ARM, Windows ARM, etc.

## What's NOT in scope (please don't PR)

- Generic vector-DB features (use FAISS / pgvector / etc.)
- Encoding logic (this library is pure search/retrieval; bring your own encoder)
- Distributed sharding (separate library)
- Major API redesigns without prior issue discussion

## Process

1. **Open an issue first** for non-trivial changes — saves both of us time
2. **Fork + branch + PR** — standard flow
3. **All tests must pass** (`cargo test` + `python examples/benchmark.py`)
4. **Add benchmark numbers** to PR description if it's a perf change
5. **License**: by contributing, you agree your contributions are licensed under Apache 2.0 (per repo LICENSE)

## Code style

- Rust: `cargo fmt` before commit
- Python: black (any reasonable formatting)
- Comments: only where the WHY is non-obvious

## Review

I (Michael) try to review PRs within 48 hours. If I haven't responded in 5 days, ping the issue thread — it might have slipped.

Performance regressions are bugs and will be rejected. Performance improvements with measurements are auto-merged after one round of review.

## Questions

Open a [GitHub Discussion](https://github.com/michaelhuo2030/hdc-neon/discussions) or email xh638@stern.nyu.edu.

Thank you.
