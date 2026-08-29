# Design Trade-offs

## Project Boundary

The current C++ code contains exact search and a basic in-memory IVF index, not a persistent vector database. The Python BM25 and WAL examples are learning exercises and are not planned as production components. Persistent segments, metadata, WAL integration, live mutations, and durable ANN indexes belong to Phase 7.

## Brute-Force Cosine Baseline

`BruteForceIndex` stores vectors in one contiguous `std::vector<float>`. This keeps every vector's components adjacent in memory and gives later ANN implementations a clear exact-search baseline.

Vectors are normalized when inserted. Queries are normalized once per search. The resulting cosine similarity is then a dot product, so each candidate score is a single linear pass over its dimensions without recomputing vector norms.

Search examines every stored vector and maintains only the current top-K results in a min-heap. This has predictable exact recall and uses O(K) extra search memory, but its linear scan makes it unsuitable as the final index at large scale. IVF and HNSW must be evaluated against it for recall and throughput.

The benchmark reports QPS, while the CTest suite validates the same index's ranking behavior, normalization, and zero-norm rejection. This separates performance measurement from correctness checking without using a different search implementation for either.

## C++ Implementation Strategy

- Use C++ for the eventual storage engine and search algorithms.
- Keep vector storage contiguous before introducing graph or inverted-list structures.
- Use deterministic random data in the benchmark so local comparisons are repeatable.
- Add focused correctness tests before treating a benchmark as a recall baseline.

## pgvector Trade-off

pgvector sacrifices some search performance and scalability to integrate directly with PostgreSQL. The trade-off provides simpler deployment, transactional consistency with relational data, and SQL support for combined relational and vector queries.

## ANN Philosophy

Exact top-K search is the correctness reference. Approximate search is useful only when it provides a meaningful speed or memory benefit while preserving acceptable recall.

### Cluster Pruning to IVF

Cluster pruning from classical IR is IVF's conceptual ancestor:

- Leaders become cluster centroids.
- Followers become vectors assigned to their nearest centroid.
- Query-time cluster selection becomes IVF's `nprobe` setting.

IVF can miss a true nearest neighbor near a cluster boundary because the relevant adjacent cluster may not be searched. Increasing `nprobe` widens the search at the cost of more work.

### Basic CPU IVF: Implemented

`IVFIndex` is a single-threaded CPU, build-then-query implementation. It is intentionally separate from product quantization, persistence, and live mutation support.

- Store unit-normalized data vectors and unit-normalized centroids; use dot product for centroid routing and candidate scoring.
- Train centroids with full-batch spherical k-means using seed 42, `nlist = 316`, mean centroid movement below `1e-4`, and a 100-iteration safety cap. Empty clusters retain their prior centroid.
- Keep one canonical contiguous vector store. Each inverted list holds only the IDs of vectors assigned to its centroid.
- Use `nprobe = 8` as the normal setting; benchmark `nprobe` values of 1, 4, 8, 16, and 32.
- Support build-then-query only. Live inserts, updates, retraining, WAL integration, and persistence are deferred.
- Use a fixed seeded synthetic dataset and query set. For each `nprobe`, report QPS and recall@10 against `BruteForceIndex`.

For 100,000 vectors, the FAISS paper's rule of thumb of approximately `sqrt(N)` suggests an initial `nlist` near 316. Larger `nprobe` improves recall by examining more lists but reduces the speed benefit.

The FAISS paper's IVFADC explanation is in Section 2, "Problem Statement." Its Section 3 is GPU architecture and GPU k-selection, so it is explicitly out of scope for this CPU implementation. Residual vectors and product quantization are complementary memory-compression techniques, but they are deferred to the dedicated PQ phase.

### Pre-Implementation Hypothesis

The exact baseline takes approximately 5.7 ms per query at 100,000 vectors and 128 dimensions (175.63 QPS). If an IVF configuration probes about 10% of balanced inverted lists, it should scan roughly 10% of the vectors after its centroid-routing step. The initial expectation is a 5-10x QPS improvement with measurable recall loss.

The latest default IVF run measured 1856.11 QPS at `nprobe = 8`, compared with 67.17 QPS for exact search in that same run: about a 27.6x speedup. Recall@10 was only 0.18 on the random-vector dataset, so the result validates the speed prediction while showing that cluster pruning needs parameter tuning and realistic-data evaluation before it can be considered high quality. The full-batch training step is CPU-intensive and resulted in substantial run-to-run timing variation; training time and query QPS should be collected under controlled conditions before making absolute performance claims.

## Storage Learning Notes

LSM trees favor write-heavy workloads through append-only segments and compaction. B+ trees favor point lookups and range scans through balanced in-place structure. The WAL exercise demonstrates that replaying complete append-only records can restore state after an interrupted write.
