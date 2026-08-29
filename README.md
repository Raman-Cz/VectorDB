# Vector DB Engine Project

A C++ vector database learning project focused on memory-aware storage, approximate nearest-neighbor search, cache-aware design, and WAL-based crash recovery.

## Current Progress

Phase 4 (ANN Algorithms) is in progress. Steps 1 and 2 are complete: the exact brute-force baseline is implemented and tested, and the KD-tree, Ball Tree, and LSH failure modes have been studied. The IVF reading is complete; implementation is deliberately deferred until the next scoped task.

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

## Current ANN Focus

The next index is a basic CPU IVF implementation, but no IVF code has been started. The relevant FAISS paper reading is the IVFADC discussion in **Section 2 (Problem Statement)**. The roadmap's reference to Section 3 is a section-number mismatch: Section 3 covers GPU architecture and GPU k-selection, which are outside this single-threaded CPU project's current scope.

The first IVF version will use k-means centroids, assign each vector to one inverted list, select the nearest `nprobe` centroids for each query, and scan only their lists using the existing exact similarity function. It will not include residual compression or product quantization; those belong to the later PQ step.

The chosen design for the first IVF version is documented in `DESIGN.md`; implementation remains intentionally deferred.

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

## Project Structure

```text
VectorDB/
  src/
    brute_force_index.hpp       Reusable exact cosine-search interface
    brute_force_index.cpp       Contiguous-vector brute-force implementation
    main.cpp                    Benchmark command-line program
    python/                     Disposable learning exercises
  tests/
    brute_force_index_test.cpp  Exact-result correctness test
  DESIGN.md                     Architectural choices and trade-offs
  notes.md                      Learning log
```
