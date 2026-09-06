# Vector DB Engine Project

A C++ vector database learning project focused on memory-aware storage, approximate nearest-neighbor search, cache-aware design, and WAL-based crash recovery.

## Current Progress

Phase 4 (ANN Algorithms) is in progress. Steps 1-3 are complete: the exact brute-force baseline, failed-approach study, and basic CPU IVF implementation are complete and benchmarked.

| Phase | Topic | Status |
| --- | --- | --- |
| 0 | Mental model: Qdrant, Weaviate, pgvector | Done |
| 1 | Information-retrieval foundations | Done |
| 2 | Linear algebra: dot product, normalization, dimensionality | Done |
| 3 | Storage internals and WAL learning exercise | Done |
| 4 | ANN algorithms: baseline, IVF, PQ, HNSW | In progress |
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

## Project Structure

```text
VectorDB/
  src/
    brute_force_index.hpp       Reusable exact cosine-search interface
    brute_force_index.cpp       Contiguous-vector brute-force implementation
    ivf_index.hpp               Basic CPU IVF interface, IVFClusterStats, and build config
    ivf_index.cpp               Spherical k-means, inverted lists, cluster stats, and nprobe search
    ivf_benchmark.cpp           QPS, cluster stats, and recall@K benchmark against exact search
    glove_loader.cpp            GloVe text-file loader
    glove_benchmark.cpp         Random held-out embedding benchmark with median QPS & cluster stats
    main.cpp                    Benchmark command-line program
    python/                     Disposable learning exercises
  tests/
    data/tiny_glove.txt         GloVe parsing test fixture
    brute_force_index_test.cpp  Exact-result correctness test
    ivf_index_test.cpp          Full-probe equivalence, stats, and validation tests
    glove_loader_test.cpp       GloVe parsing test using a tiny fixture
  DESIGN.md                     Architectural choices and trade-offs
  notes.md                      Learning log
```
