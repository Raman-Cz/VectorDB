# Design Trade-offs

## Project Boundary

The current C++ code is an exact-search benchmark, not a persistent vector database. The Python BM25 and WAL examples are learning exercises and are not planned as production components. Persistent segments, metadata, WAL integration, and ANN indexes belong to Phase 7.

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

## Storage Learning Notes

LSM trees favor write-heavy workloads through append-only segments and compaction. B+ trees favor point lookups and range scans through balanced in-place structure. The WAL exercise demonstrates that replaying complete append-only records can restore state after an interrupted write.
