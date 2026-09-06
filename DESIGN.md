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
- Compute cluster size statistics (`IVFClusterStats`: min, max, mean, stddev, empty clusters) post-build to diagnose cluster balance quality.
- Benchmark fine-grained `nprobe` values `{1, 4, 8, 10, 12, 14, 16, 32}` to capture the knee of the recall-vs-speed curve.
- Support build-then-query only. Live inserts, updates, retraining, WAL integration, and persistence are deferred.
- Sample held-out query vectors uniformly at random from the embedding file using seeded PRNG (`std::mt19937(42)`) to eliminate vocabulary frequency bias. Report median QPS across repeated runs and recall@10 against `BruteForceIndex`.

For 100,000 vectors, the FAISS paper's rule of thumb of approximately `sqrt(N)` suggests an initial `nlist` near 316. Larger `nprobe` improves recall by examining more lists but reduces the speed benefit.

The FAISS paper's IVFADC explanation is in Section 2, "Problem Statement." Its Section 3 is GPU architecture and GPU k-selection, so it is explicitly out of scope for this CPU implementation. Residual vectors and product quantization are complementary memory-compression techniques, but they are deferred to the dedicated PQ phase.

### Pre-Implementation Hypothesis

The exact baseline takes approximately 5.7 ms per query at 100,000 vectors and 128 dimensions (175.63 QPS). If an IVF configuration probes about 10% of balanced inverted lists, it should scan roughly 10% of the vectors after its centroid-routing step. The initial expectation is a 5-10x QPS improvement with measurable recall loss.

The synthetic IVF run measured 1106.99 QPS at `nprobe = 8`, compared with 52.49 QPS for exact search in that same run: about a 21.1x speedup. Recall@10 was 0.18 on the random-vector dataset. Cluster sizes were uniform (`min: 271, max: 359, stddev: 14.48`). This confirms that random uniform vectors do not form natural semantic clusters.

On the GloVe-50 held-out split (500 random held-out queries), `nprobe = 8` reached 0.84 recall@10 at 2755.74 median QPS, `nprobe = 14` reached 0.91 recall@10 at 1682.95 median QPS, and `nprobe = 32` reached 0.97 recall@10 at 718.73 median QPS. The same run measured brute force at 246.36 median QPS.

Real embedding clusters showed significant size variance (`min: 72, max: 934, stddev: 101.09`), demonstrating that semantic embeddings produce dense and sparse vector spaces that benefit from cluster routing.

### `nlist` Parameter Sweep Analysis & Effective Search Fraction

Sweeping $N_{list} \in \{100, 316, 1000, 2000\}$ on 100,000 GloVe embeddings highlights the joint mechanics of cluster granularity ($N_{list}$) and probing depth ($N_{probe}$):

- **Effective Search Fraction**: Recall is primarily governed by the fraction of the dataset searched ($\frac{N_{probe}}{N_{list}}$). At fixed $N_{probe}=8$, $N_{list}=100$ searches $8\%$ of the dataset (achieving $0.91$ recall), while $N_{list}=2000$ searches only $0.4\%$ of the dataset (achieving $0.70$ recall).
- **Target Recall Efficiency**: When holding target recall constant, higher $N_{list}$ with smaller clusters yields superior QPS. For example, at $N_{list}=2000, N_{probe}=32$ ($\text{fraction} = 1.6\%$), the index achieves **0.89 recall@10** at **3,521 QPS**, outperforming $N_{list}=316, N_{probe}=8$ ($\text{fraction} = 2.5\%$) which achieves **0.84 recall@10** at **2,755 QPS**.
- **Practical Balance ($\sqrt{N} \approx 316$)**: While higher $N_{list}$ increases QPS efficiency at equal recall, it increases centroid routing overhead and training time ($226$s vs $86$s). $\sqrt{N} \approx 316$ represents a pragmatic engineering baseline balancing training speed, memory footprint, and query performance.

## Storage Learning Notes

LSM trees favor write-heavy workloads through append-only segments and compaction. B+ trees favor point lookups and range scans through balanced in-place structure. The WAL exercise demonstrates that replaying complete append-only records can restore state after an interrupted write.
