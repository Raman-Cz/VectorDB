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

`glove_benchmark` evaluates IVF on a deterministic held-out split from a GloVe text file. The first `index_count` embeddings form the index; the following `query_count` embeddings are held out as queries. It reports every QPS sample, the median across repeated runs, and recall@K against `BruteForceIndex`.

```powershell
.\build\Release\glove_benchmark.exe data\glove.6B.50d.txt
```

Optional arguments are `dimensions index_count query_count top_k nlist repetitions`:

```powershell
.\build\Release\glove_benchmark.exe data\glove.6B.50d.txt 50 100000 100 10 316 3
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

On 2026-08-29, the default IVF benchmark used 100,000 random 128-dimensional vectors, 100 queries, top-K 10, and `nlist = 316`. The latest run took 293.57 seconds to train centroids; that one-time training cost is separate from query QPS.

| `nprobe` | QPS | recall@10 |
| --- | ---: | ---: |
| 1 | 9717.13 | 0.04 |
| 4 | 3399.18 | 0.10 |
| 8 | 1856.11 | 0.18 |
| 16 | 927.17 | 0.27 |
| 32 | 488.07 | 0.40 |

The same run measured brute-force search at 67.17 QPS. IVF at `nprobe = 8` is about 27.6x faster, but its 0.18 recall@10 shows why QPS must always be evaluated alongside recall. The prolonged CPU-bound training step affected system performance between runs, so use these values as a reproducible sample rather than stable hardware limits.

### Recorded GloVe Run

On 2026-08-31, the held-out GloVe-50 benchmark indexed 100,000 embeddings and queried the next 100 embeddings in the file. It used `nlist = 316` and three timing repetitions per configuration; reported QPS is the median. IVF training took 74.59 seconds and brute-force median QPS was 534.92.

| `nprobe` | median QPS | recall@10 |
| --- | ---: | ---: |
| 1 | 39019.82 | 0.48 |
| 4 | 12068.55 | 0.75 |
| 8 | 3376.47 | 0.85 |
| 16 | 1291.74 | 0.93 |
| 32 | 911.46 | 0.98 |

The real embeddings make IVF's cluster routing meaningful: `nprobe = 8` provides about 6.3x the brute-force median QPS at 0.85 recall@10, while `nprobe = 32` reaches 0.98 recall@10 at about 1.7x brute-force QPS.

## Project Structure

```text
VectorDB/
  src/
    brute_force_index.hpp       Reusable exact cosine-search interface
    brute_force_index.cpp       Contiguous-vector brute-force implementation
    ivf_index.hpp               Basic CPU IVF interface and build configuration
    ivf_index.cpp               Spherical k-means, inverted lists, and nprobe search
    ivf_benchmark.cpp           QPS and recall@K benchmark against exact search
    glove_loader.cpp            GloVe text-file loader
    glove_benchmark.cpp         Held-out embedding benchmark with median QPS
    main.cpp                    Benchmark command-line program
    python/                     Disposable learning exercises
  tests/
    brute_force_index_test.cpp  Exact-result correctness test
    ivf_index_test.cpp          Full-probe equivalence and validation tests
    glove_loader_test.cpp       GloVe parsing test using a tiny fixture
  DESIGN.md                     Architectural choices and trade-offs
  notes.md                      Learning log
```
