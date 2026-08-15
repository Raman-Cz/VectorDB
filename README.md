# Vector DB Engine Project

A C++ vector database learning project focused on memory-aware storage, approximate nearest-neighbor search, cache-aware design, and WAL-based crash recovery.

## Current Progress

Phase 4 (ANN Algorithms) is in progress. Step 1, the exact brute-force cosine-search baseline, is implemented and tested; it will be used to measure recall and speed improvements for IVF and HNSW.

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
