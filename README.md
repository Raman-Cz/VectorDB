# Vector DB Engine Project

A C++ vector database learning project focused on memory-aware storage, approximate nearest-neighbor search, cache-aware design, and WAL-based crash recovery.

## Current Progress

Phase 4 (ANN Algorithms) is in progress. Steps 1-4 are complete: the exact brute-force baseline, failed-approach study, basic CPU IVF implementation, and Product Quantization (PQ) vector compression are complete and benchmarked.

| Phase | Topic | Status |
| --- | --- | --- |
| 0 | Mental model: Qdrant, Weaviate, pgvector | Done |
| 1 | Information-retrieval foundations | Done |
| 2 | Linear algebra: dot product, normalization, dimensionality | Done |
| 3 | Storage internals and WAL learning exercise | Done |
| 4 | ANN algorithms: baseline, IVF, PQ, HNSW | In progress (PQ complete) |
| 5 | Read FAISS source | Planned |
| 6 | Memory and cache performance model | Planned |
| 7 | Build the persistent vector database | Planned |

The Python BM25 and WAL code are disposable learning exercises. Production-oriented database work remains planned for Phase 7.

## Basic CPU IVF

`IVFIndex` trains unit-normalized centroids with spherical k-means, assigns vector IDs to centroid-owned inverted lists, and probes the nearest lists at query time. It stores vectors once in a canonical contiguous array and uses the same normalized dot-product similarity as `BruteForceIndex`.

This is a CPU-only, build-then-query index. It intentionally excludes product quantization, persistence, live inserts, updates, and retraining. The relevant FAISS reading is the IVFADC discussion in **Section 2 (Problem Statement)**; GPU-focused Section 3 remains out of scope.

Run the IVF benchmark, which reports QPS and recall@K against exact search:

```powershell
.\build\Release\ivf_benchmark.exe
```

Optional arguments are `vector_count dimensions query_count top_k nlist`:

```powershell
.\build\Release\ivf_benchmark.exe 100000 128 100 10 316
```

## Real-Embedding Evaluation

`glove_benchmark` evaluates IVF on a deterministic held-out split from a GloVe text file. The first `index_count` embeddings form the index; held-out query vectors are sampled uniformly at random across the remaining embeddings to eliminate vocabulary frequency bias. It reports cluster size statistics (`min`, `max`, `mean`, `stddev`, `empty`), every QPS sample, the median across repeated runs, and recall@K against `BruteForceIndex`.

```powershell
.\build\Release\glove_benchmark.exe data\glove.6B.50d.txt
```

Optional arguments are `dimensions index_count query_count top_k nlist repetitions`:

```powershell
.\build\Release\glove_benchmark.exe data\glove.6B.50d.txt 50 100000 500 10 316 3
```

The dataset directory is ignored by Git. Download and extract the GloVe 6B archive locally before running this command.

## Brute-Force Baseline

`vector_db` creates random vectors, normalizes them on insertion, performs exact top-K cosine search, and reports query throughput. Since all indexed vectors and each query are normalized, cosine similarity is computed as a dot product.

Vectors are stored in one flat contiguous `std::vector<float>` of `vector_count * dimensions` values, rather than as separately allocated vector rows. The benchmark reports performance only; the CTest case validates the same `BruteForceIndex` implementation for normalized ranking, sorted top-K results, and zero-norm input rejection.

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\vector_db.exe
```

Default benchmark parameters are 100,000 vectors, 128 dimensions, 100 queries, and top-K 10. Override them with:

```powershell
.\build\Release\vector_db.exe [vector_count] [dimensions] [query_count] [top_k]
```

Run the correctness test with:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

### Recorded Baseline

One local Release run on 2026-08-15 completed 100 queries against 100,000 random 128-dimensional vectors with top-K 10 in 0.57 seconds: **175.63 QPS**. Throughput will vary by hardware and background load; rerun this command before comparing a future IVF or HNSW implementation.

### Recorded IVF Run

On 2026-09-06, the default synthetic IVF benchmark used 100,000 random 128-dimensional vectors, 100 queries, top-K 10, and `nlist = 316`. Centroid training took 305.45 seconds. Cluster sizes were nearly uniform across centroids (`min: 271, max: 359, mean: 316.46, stddev: 14.48, empty: 0`).

| `nprobe` | QPS | recall@10 |
| --- | ---: | ---: |
| 1 | 6380.52 | 0.04 |
| 4 | 1955.88 | 0.10 |
| 8 | 1106.99 | 0.18 |
| 10 | 933.12 | 0.21 |
| 12 | 725.12 | 0.23 |
| 14 | 795.50 | 0.26 |
| 16 | 721.24 | 0.27 |
| 32 | 341.27 | 0.40 |

The same run measured brute-force search at 52.49 QPS. IVF at `nprobe = 8` is about 21.1x faster, but its 0.18 recall@10 shows why synthetic random vectors produce poor cluster partitioning compared to real embeddings.

### Recorded GloVe Run

On 2026-09-06, the GloVe-50 benchmark indexed 100,000 embeddings and evaluated 500 held-out query vectors sampled uniformly at random across the vocabulary pool. It used `nlist = 316` and three timing repetitions per configuration; reported QPS is the median. Centroid training took 85.97 seconds and brute-force median QPS was 246.36.

Cluster sizes reflected true real-world embedding skew (`min: 72, max: 934, mean: 316.46, stddev: 101.09, empty: 0`).

| `nprobe` | median QPS | recall@10 |
| --- | ---: | ---: |
| 1 | 19204.92 | 0.45 |
| 4 | 5454.62 | 0.73 |
| 8 | 2755.74 | 0.84 |
| 10 | 2328.27 | 0.87 |
| 12 | 1907.47 | 0.89 |
| 14 | 1682.95 | 0.91 |
| 16 | 1430.88 | 0.92 |
| 32 | 718.73 | 0.97 |

Real semantic embeddings give IVF meaningful cluster routing: `nprobe = 8` provides about 11.2x brute-force throughput at 0.84 recall@10; `nprobe = 14` reaches 0.91 recall@10 at 6.8x brute-force speed; and `nprobe = 32` achieves 0.97 recall@10.

### Coarse Cluster Sweep (`nlist`) Comparison

Sweeping $N_{list} \in \{100, 316, 1000, 2000\}$ on GloVe-50 demonstrates the trade-off between cluster size, search throughput, and recall@10:

| `nlist` | Build Time | Cluster Mean Size | Cluster StdDev | `nprobe=8` QPS | `nprobe=8` Recall | `nprobe=14` QPS | `nprobe=14` Recall | `nprobe=32` QPS | `nprobe=32` Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **100** | 15.18s | 1000.0 | 278.8 | 1795.97 | 0.91 | 893.58 | 0.96 | 401.16 | 0.99 |
| **316** ($\approx \sqrt{N}$) | 85.97s | 316.5 | 101.1 | 2755.74 | 0.84 | 1682.95 | 0.91 | 718.73 | 0.97 |
| **1000** | 171.31s | 100.0 | 38.7 | 7623.52 | 0.74 | 5241.04 | 0.83 | 2657.79 | 0.92 |
| **2000** | 225.78s | 50.0 | 21.7 | 8595.14 | 0.70 | 6346.38 | 0.78 | 3521.20 | 0.89 |

## Product Quantization (PQ) Vector Compression

`ProductQuantizer` and `PQIndex` implement lossy vector compression by partitioning $D$-dimensional vectors into $M$ sub-vectors and quantizing each sub-space independently into $K^*=256$ centroids. Each vector is stored as $M$ bytes (1 byte per sub-space codebook index), and queries run fast **Asymmetric Distance Computation (ADC)** via pre-computed lookup tables.

```powershell
.\build\Release\pq_benchmark.exe data\glove.6B.50d.txt
```

### Recorded PQ Run

On 2026-09-14, `pq_benchmark` evaluated 100,000 GloVe-50 vectors (uncompressed size 200 bytes per vector) across 500 randomly sampled held-out queries. All vectors are unit-normalized ($L_2 = 1.0$) to align squared Euclidean ADC distance with exact cosine ranking. Brute-force median QPS was 517.89.

| $M$ (Sub-vectors) | Bytes/Vector | Compression | Train Time | median QPS | recall@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **5** | 5 bytes | **40.00x** | 22.73s | 2919.31 | 0.26 |
| **10** | 10 bytes | **20.00x** | 31.44s | 1915.56 | 0.52 |
| **25** | 25 bytes | **8.00x** | 49.06s | 1076.18 | **0.87** |

At $M=5$, PQ achieves a **40x memory compression ratio** (shrinking 200-byte vectors to 5 bytes) and 2,919 QPS. At $M=25$ (8x compression), recall@10 reaches **0.87** with 1,076 QPS.

## Inverted File with Product Quantization (IVFPQ)

`IVFPQIndex` combines coarse cluster pruning (IVF) with vector compression (PQ). Vectors are assigned to coarse clusters, and their residual vectors ($r = v - c$) are quantized using sub-space codebooks into $M$ byte codes per vector. Scoring uses exact dot-product linear expansion: $\text{Query} \cdot v = (\text{Query} \cdot c) + \text{AsymmetricDotProduct}(\text{codes}, \text{table})$.

```powershell
.\build\Release\ivfpq_benchmark.exe data\glove.6B.50d.txt
```

### Recorded IVFPQ Run

On 2026-09-20, `ivfpq_benchmark` evaluated 100,000 GloVe-50 vectors across 500 randomly sampled held-out queries with $N_{list} = 316$. Brute-force median QPS was 516.34.

| $M$ (Sub-vecs) | Bytes/Vector | Compression | $N_{probe}$ | Build Time | median QPS | QPS Range | recall@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **10** | 10 B | **20.00x** | 1 | 54.23s | **32,838.13** | 30,132-33,114 | 0.3262 |
| **10** | 10 B | **20.00x** | 8 | 54.23s | **15,941.54** | 15,549-16,154 | 0.4910 |
| **10** | 10 B | **20.00x** | 32 | 54.23s | **5,545.57** | 5,468-5,553 | 0.5110 |
| **25** | 25 B | **8.00x** | 1 | 62.28s | **44,092.49** | 41,986-44,896 | 0.4252 |
| **25** | 25 B | **8.00x** | 8 | 62.28s | **14,588.49** | 14,473-14,769 | **0.7712** |
| **25** | 25 B | **8.00x** | 32 | 62.28s | **5,226.70** | 5,093-5,272 | **0.8536** |

### Index Comparison Overview

| Index Strategy | Memory / Vector | $100\text{k}$ Index Size | Throughput (QPS) | Recall@10 | Key Trade-off |
| --- | ---: | ---: | ---: | ---: | --- |
| **Brute Force** | 200 B | 20.0 MB | 516.34 | 1.0000 | Ground truth reference; scans 100% of data |
| **IVF Alone** ($N_{probe}=8$) | 200 B | 20.0 MB | 2,755.74 | 0.8400 | Fast search; no vector compression |
| **PQ Alone** ($M=25$) | 25 B | 2.5 MB | 1,076.18 | 0.8700 | 8x memory reduction; exhaustive scan |
| **IVFPQ** ($M=25, N_{probe}=8$) | **25 B** | **2.5 MB** | **14,588.49** | **0.7712** | **8x memory reduction + 28x throughput speedup** |
| **IVFPQ** ($M=25, N_{probe}=32$) | **25 B** | **2.5 MB** | **5,226.70** | **0.8536** | **8x memory reduction + 10x throughput speedup at high recall** |

## Project Structure

```text
VectorDB/
  src/
    brute_force_index.hpp       Reusable exact cosine-search interface
    brute_force_index.cpp       Contiguous-vector brute-force implementation
    ivf_index.hpp               Basic CPU IVF interface, IVFClusterStats, and build config
    ivf_index.cpp               Spherical k-means, inverted lists, cluster stats, and nprobe search
    ivf_benchmark.cpp           QPS, cluster stats, and recall@K benchmark against exact search
    product_quantizer.hpp       Sub-vector partitioning, K-means codebooks, ADC & dot-product tables
    product_quantizer.cpp       Codebook training, 1-byte encoding/decoding, ADC & dot-product lookups
    pq_index.hpp                Compressed in-memory index storing M bytes per vector
    pq_index.cpp                PQ compressed search using pre-computed ADC lookup tables
    pq_benchmark.cpp             Memory footprint, QPS, and recall@10 benchmark harness
    ivfpq_index.hpp             IVFPQ interface (coarse IVF + residual PQ)
    ivfpq_index.cpp             Inverted lists of PQ residual codes & dot-product residual scoring
    ivfpq_benchmark.cpp           IVFPQ memory, QPS spread, and recall@10 benchmark harness
    glove_loader.cpp            GloVe text-file loader
    glove_benchmark.cpp         Random held-out embedding benchmark with median QPS & cluster stats
    main.cpp                    Benchmark command-line program
    python/                     Disposable learning exercises
  tests/
    data/tiny_glove.txt         GloVe parsing test fixture
    brute_force_index_test.cpp  Exact-result correctness test
    ivf_index_test.cpp          Full-probe equivalence, stats, and validation tests
    glove_loader_test.cpp       GloVe parsing test using a tiny fixture
    product_quantizer_test.cpp  Sub-vector partitioning, encoding/decoding, and ADC/dot-product tests
    ivfpq_index_test.cpp        IVFPQ build, cluster stats, memory accounting & full-probe tests
  DESIGN.md                     Architectural choices and trade-offs
  notes.md                      Learning log
```
