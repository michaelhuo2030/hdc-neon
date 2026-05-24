// Copyright 2026 Michael (Xiaojie) Huo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

//! HDC NEON SIMD ops — fast Hamming distance + int8 dot product + fp32 dot.
//!
//! On ARM64 (Apple M-series):
//!   - `u64::count_ones()` → `vcntq_u8 + addv` (NEON popcount path) — for bipolar Hamming
//!   - i32 accumulation of `i16` products → uses `vmlal_s16` (multiply-accumulate widen) — for int8 dot
//!   - f32 fma → uses `vfmaq_f32` — for fp32 dot (Fourier real+imag)
//!
//! API:
//!   - `hamming_topk(docs_packed: u64[N, W], query: u64[W], k) -> u64[k]`
//!   - `hamming_all(docs_packed: u64[N, W], query: u64[W]) -> u32[N]`
//!   - `int8_dot_topk(docs: i8[N, D], query: i8[D], k) -> u64[k]`  (NEW — multi-bit)
//!   - `f32_dot_topk(docs: f32[N, D], query: f32[D], k) -> u64[k]`  (NEW — Fourier real/imag)
//!   - `fourier_topk(doc_real: f32[N, D], doc_imag: f32[N, D], q_real: f32[D], q_imag: f32[D], k) -> u64[k]`

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Hamming distance between two packed bit-vectors (u64 slices).
#[inline]
fn hamming_u64(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut h: u32 = 0;
    for i in 0..a.len() {
        h += (a[i] ^ b[i]).count_ones();
    }
    h
}

#[pyfunction]
fn hamming_all<'py>(
    py: Python<'py>,
    docs: PyReadonlyArray2<u64>,
    query: PyReadonlyArray1<u64>,
) -> Bound<'py, PyArray1<u32>> {
    let docs = docs.as_array();
    let q = query.as_slice().unwrap();
    let n = docs.shape()[0];
    let w = docs.shape()[1];
    assert_eq!(w, q.len(), "docs width and query length must match");

    let docs_slice = docs.as_slice().unwrap();
    let out: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|i| hamming_u64(&docs_slice[i * w..(i + 1) * w], q))
        .collect();
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn hamming_topk<'py>(
    py: Python<'py>,
    docs: PyReadonlyArray2<u64>,
    query: PyReadonlyArray1<u64>,
    k: usize,
) -> Bound<'py, PyArray1<u64>> {
    let docs = docs.as_array();
    let q = query.as_slice().unwrap();
    let n = docs.shape()[0];
    let w = docs.shape()[1];
    assert_eq!(w, q.len(), "docs width and query length must match");

    let docs_slice = docs.as_slice().unwrap();
    // Parallel-compute distances
    let dists: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|i| hamming_u64(&docs_slice[i * w..(i + 1) * w], q))
        .collect();

    // Top-k by smallest distance
    let mut indexed: Vec<(u32, u64)> = dists
        .iter()
        .enumerate()
        .map(|(i, &d)| (d, i as u64))
        .collect();
    indexed.select_nth_unstable_by_key(k, |x| x.0);
    let mut top: Vec<(u32, u64)> = indexed[..k].to_vec();
    top.sort_by_key(|x| x.0);
    let out: Vec<u64> = top.into_iter().map(|x| x.1).collect();
    out.into_pyarray_bound(py)
}

// ===== int8 dot product (for multi-bit HDC cosine) =====

#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: i32 = 0;
    for i in 0..a.len() {
        acc += (a[i] as i32) * (b[i] as i32);
    }
    acc
}

#[pyfunction]
fn int8_dot_topk<'py>(
    py: Python<'py>,
    docs: PyReadonlyArray2<i8>,
    query: PyReadonlyArray1<i8>,
    k: usize,
) -> Bound<'py, PyArray1<u64>> {
    let docs = docs.as_array();
    let q = query.as_slice().unwrap();
    let n = docs.shape()[0];
    let d = docs.shape()[1];
    assert_eq!(d, q.len());
    let docs_slice = docs.as_slice().unwrap();
    // Parallel: compute dot products
    let dots: Vec<i32> = (0..n)
        .into_par_iter()
        .map(|i| dot_i8(&docs_slice[i * d..(i + 1) * d], q))
        .collect();
    // Top-k by LARGEST dot (similarity, not distance) — negate for argpartition-style
    let mut indexed: Vec<(i32, u64)> = dots
        .iter()
        .enumerate()
        .map(|(i, &x)| (-x, i as u64))
        .collect();
    indexed.select_nth_unstable_by_key(k, |x| x.0);
    let mut top: Vec<(i32, u64)> = indexed[..k].to_vec();
    top.sort_by_key(|x| x.0);
    let out: Vec<u64> = top.into_iter().map(|x| x.1).collect();
    out.into_pyarray_bound(py)
}

// ===== fp32 dot product (for Fourier real/imag) =====

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: f32 = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[pyfunction]
fn f32_dot_topk<'py>(
    py: Python<'py>,
    docs: PyReadonlyArray2<f32>,
    query: PyReadonlyArray1<f32>,
    k: usize,
) -> Bound<'py, PyArray1<u64>> {
    let docs = docs.as_array();
    let q = query.as_slice().unwrap();
    let n = docs.shape()[0];
    let d = docs.shape()[1];
    assert_eq!(d, q.len());
    let docs_slice = docs.as_slice().unwrap();
    let dots: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| dot_f32(&docs_slice[i * d..(i + 1) * d], q))
        .collect();
    let mut indexed: Vec<(f32, u64)> = dots
        .iter()
        .enumerate()
        .map(|(i, &x)| (-x, i as u64))
        .collect();
    indexed.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut top: Vec<(f32, u64)> = indexed[..k].to_vec();
    top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let out: Vec<u64> = top.into_iter().map(|x| x.1).collect();
    out.into_pyarray_bound(py)
}

// ===== Fourier topk: real_dot + imag_dot fused =====

#[pyfunction]
fn fourier_topk<'py>(
    py: Python<'py>,
    doc_real: PyReadonlyArray2<f32>,
    doc_imag: PyReadonlyArray2<f32>,
    q_real: PyReadonlyArray1<f32>,
    q_imag: PyReadonlyArray1<f32>,
    k: usize,
) -> Bound<'py, PyArray1<u64>> {
    let dr = doc_real.as_array();
    let di = doc_imag.as_array();
    let qr = q_real.as_slice().unwrap();
    let qi = q_imag.as_slice().unwrap();
    let n = dr.shape()[0];
    let d = dr.shape()[1];
    let dr_slice = dr.as_slice().unwrap();
    let di_slice = di.as_slice().unwrap();
    // sim_i = dot(d_real[i], q_real) + dot(d_imag[i], q_imag)
    let sims: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let real_part = dot_f32(&dr_slice[i * d..(i + 1) * d], qr);
            let imag_part = dot_f32(&di_slice[i * d..(i + 1) * d], qi);
            real_part + imag_part
        })
        .collect();
    let mut indexed: Vec<(f32, u64)> = sims
        .iter()
        .enumerate()
        .map(|(i, &x)| (-x, i as u64))
        .collect();
    indexed.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut top: Vec<(f32, u64)> = indexed[..k].to_vec();
    top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let out: Vec<u64> = top.into_iter().map(|x| x.1).collect();
    out.into_pyarray_bound(py)
}

// ===== Batched query Hamming: B queries vs N docs in one call =====

#[pyfunction]
fn hamming_topk_batched<'py>(
    py: Python<'py>,
    docs: PyReadonlyArray2<u64>,
    queries: PyReadonlyArray2<u64>,
    k: usize,
) -> Bound<'py, PyArray1<u64>> {
    // Returns a flat (nq * k) u64 array; Python reshapes to (nq, k).
    let docs = docs.as_array();
    let queries = queries.as_array();
    let n = docs.shape()[0];
    let w = docs.shape()[1];
    let nq = queries.shape()[0];
    assert_eq!(w, queries.shape()[1]);
    let docs_slice = docs.as_slice().unwrap();
    let queries_slice = queries.as_slice().unwrap();

    let mut out = vec![0u64; nq * k];
    for q_idx in 0..nq {
        let q = &queries_slice[q_idx * w..(q_idx + 1) * w];
        let dists: Vec<u32> = (0..n)
            .into_par_iter()
            .map(|i| hamming_u64(&docs_slice[i * w..(i + 1) * w], q))
            .collect();
        let mut indexed: Vec<(u32, u64)> = dists.iter().enumerate().map(|(i, &d)| (d, i as u64)).collect();
        indexed.select_nth_unstable_by_key(k, |x| x.0);
        let mut top: Vec<(u32, u64)> = indexed[..k].to_vec();
        top.sort_by_key(|x| x.0);
        for (j, (_, idx)) in top.into_iter().enumerate() {
            out[q_idx * k + j] = idx;
        }
    }
    out.into_pyarray_bound(py)
}

// ===== Branch-and-bound top-K: early-terminate per-doc =====

#[pyfunction]
fn hamming_topk_bnb<'py>(
    py: Python<'py>,
    docs: PyReadonlyArray2<u64>,
    query: PyReadonlyArray1<u64>,
    k: usize,
) -> Bound<'py, PyArray1<u64>> {
    let docs = docs.as_array();
    let q = query.as_slice().unwrap();
    let n = docs.shape()[0];
    let w = docs.shape()[1];
    let docs_slice = docs.as_slice().unwrap();

    // Pass 1: get a rough top-K via the first W/4 words only (cheap)
    let prefix = (w / 4).max(1);
    let prefix_dists: Vec<u32> = (0..n).into_par_iter()
        .map(|i| {
            let row = &docs_slice[i * w..i * w + prefix];
            let q_prefix = &q[..prefix];
            hamming_u64(row, q_prefix)
        }).collect();
    let mut prefix_sorted: Vec<(u32, u64)> = prefix_dists.iter().enumerate().map(|(i, &d)| (d, i as u64)).collect();
    prefix_sorted.select_nth_unstable_by_key(k, |x| x.0);
    let initial_threshold: u32 = prefix_sorted[..k].iter().map(|x| x.0).max().unwrap();
    // Project: full Hamming will be ≤ prefix_hamming + (w - prefix) * 64
    let projection_bound = initial_threshold + ((w - prefix) as u32) * 64;

    // Pass 2: full hamming, but skip docs whose prefix already exceeds the bound
    let full_dists: Vec<u32> = (0..n).into_par_iter()
        .map(|i| {
            // Skip if prefix already exceeds projection bound
            if prefix_dists[i] > projection_bound {
                return u32::MAX;
            }
            let mut h = prefix_dists[i];
            let row = &docs_slice[i * w..(i + 1) * w];
            for j in prefix..w {
                h += (row[j] ^ q[j]).count_ones();
            }
            h
        }).collect();

    let mut indexed: Vec<(u32, u64)> = full_dists.iter().enumerate().map(|(i, &d)| (d, i as u64)).collect();
    indexed.select_nth_unstable_by_key(k, |x| x.0);
    let mut top: Vec<(u32, u64)> = indexed[..k].to_vec();
    top.sort_by_key(|x| x.0);
    let out: Vec<u64> = top.into_iter().map(|x| x.1).collect();
    out.into_pyarray_bound(py)
}

// ===== Tiled NEON Hamming: process docs in L1-cache-fitting tiles =====
// M-series L1d cache is ~192 KB/core. At 1.25 KB per doc (D=10K bit-packed),
// that's ~150 docs per tile. We pick 128 docs/tile as a power of 2.

const TILE_DOCS: usize = 128;

#[pyfunction]
fn hamming_topk_tiled<'py>(
    py: Python<'py>,
    docs: PyReadonlyArray2<u64>,
    query: PyReadonlyArray1<u64>,
    k: usize,
) -> Bound<'py, PyArray1<u64>> {
    let docs = docs.as_array();
    let q = query.as_slice().unwrap();
    let n = docs.shape()[0];
    let w = docs.shape()[1];
    let docs_slice = docs.as_slice().unwrap();

    // Tile-and-parallelize: each rayon task takes a TILE_DOCS-sized chunk
    let num_tiles = (n + TILE_DOCS - 1) / TILE_DOCS;
    let dists: Vec<u32> = (0..num_tiles)
        .into_par_iter()
        .flat_map(|tile_idx| {
            let start = tile_idx * TILE_DOCS;
            let end = std::cmp::min(start + TILE_DOCS, n);
            // Compute Hamming for docs in this tile — Q stays in registers, docs streamed once
            let mut tile_dists = Vec::with_capacity(end - start);
            for i in start..end {
                let row = &docs_slice[i * w..(i + 1) * w];
                let mut h: u32 = 0;
                // Unrolled inner loop for compiler-friendly NEON vectorization
                let mut j = 0;
                while j + 4 <= w {
                    h += (row[j] ^ q[j]).count_ones();
                    h += (row[j + 1] ^ q[j + 1]).count_ones();
                    h += (row[j + 2] ^ q[j + 2]).count_ones();
                    h += (row[j + 3] ^ q[j + 3]).count_ones();
                    j += 4;
                }
                while j < w {
                    h += (row[j] ^ q[j]).count_ones();
                    j += 1;
                }
                tile_dists.push(h);
            }
            tile_dists
        })
        .collect();

    let mut indexed: Vec<(u32, u64)> = dists.iter().enumerate().map(|(i, &d)| (d, i as u64)).collect();
    indexed.select_nth_unstable_by_key(k, |x| x.0);
    let mut top: Vec<(u32, u64)> = indexed[..k].to_vec();
    top.sort_by_key(|x| x.0);
    let out: Vec<u64> = top.into_iter().map(|x| x.1).collect();
    out.into_pyarray_bound(py)
}

// ===== Hierarchical coarse+fine top-K =====
// Pass 1: coarse Hamming on small-D HVs to filter top-M candidates
// Pass 2: full-D Hamming only on those candidates

#[pyfunction]
fn hamming_topk_hierarchical<'py>(
    py: Python<'py>,
    docs_coarse: PyReadonlyArray2<u64>,
    q_coarse: PyReadonlyArray1<u64>,
    docs_fine: PyReadonlyArray2<u64>,
    q_fine: PyReadonlyArray1<u64>,
    k: usize,
    m_candidates: usize,
) -> Bound<'py, PyArray1<u64>> {
    let dc = docs_coarse.as_array();
    let qc = q_coarse.as_slice().unwrap();
    let df = docs_fine.as_array();
    let qf = q_fine.as_slice().unwrap();
    let n = dc.shape()[0];
    let wc = dc.shape()[1];
    let wf = df.shape()[1];
    let dc_slice = dc.as_slice().unwrap();
    let df_slice = df.as_slice().unwrap();

    // Pass 1: coarse on all N docs
    let coarse_dists: Vec<u32> = (0..n).into_par_iter()
        .map(|i| hamming_u64(&dc_slice[i * wc..(i + 1) * wc], qc))
        .collect();
    let mut indexed: Vec<(u32, usize)> = coarse_dists.iter().enumerate().map(|(i, &d)| (d, i)).collect();
    indexed.select_nth_unstable_by_key(m_candidates, |x| x.0);
    let candidates: Vec<usize> = indexed[..m_candidates].iter().map(|x| x.1).collect();

    // Pass 2: fine on M candidates
    let fine_dists: Vec<(u32, u64)> = candidates.par_iter()
        .map(|&i| (hamming_u64(&df_slice[i * wf..(i + 1) * wf], qf), i as u64))
        .collect();
    let mut fine_sorted = fine_dists;
    fine_sorted.select_nth_unstable_by_key(k, |x| x.0);
    let mut top: Vec<(u32, u64)> = fine_sorted[..k].to_vec();
    top.sort_by_key(|x| x.0);
    let out: Vec<u64> = top.into_iter().map(|x| x.1).collect();
    out.into_pyarray_bound(py)
}

#[pymodule]
fn hdc_neon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hamming_all, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_topk, m)?)?;
    m.add_function(wrap_pyfunction!(int8_dot_topk, m)?)?;
    m.add_function(wrap_pyfunction!(f32_dot_topk, m)?)?;
    m.add_function(wrap_pyfunction!(fourier_topk, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_topk_batched, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_topk_bnb, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_topk_hierarchical, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_topk_tiled, m)?)?;
    Ok(())
}
