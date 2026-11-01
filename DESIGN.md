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

### Product Quantization (PQ): Implemented

`ProductQuantizer` and `PQIndex` implement lossy vector compression to address vector storage memory bottlenecks.

- **Sub-Vector Division**: $D$-dimensional vectors are split into $M$ sub-vectors of dimension $D_{sub} = D / M$.
- **Sub-Space Codebooks**: Independent K-Means models learn $K^* = 256$ centroids per sub-space.
- **Compact Byte Encoding**: Each vector is compressed into $M$ bytes (1 byte per sub-space codebook index), achieving up to $40\times$ memory reduction ($200$ bytes $\rightarrow$ $5$ bytes for 50D vectors at $M=5$).
- **Asymmetric Distance Computation (ADC)**: Pre-computes a 2D distance lookup table ($M \times 256$ floats) per query, enabling fast approximate distance calculations via $M$ byte lookups and additions per candidate vector.
- **Metric Normalization Alignment**: To ensure compatibility with exact cosine ground truth, `PQIndex` enforces $L_2 = 1.0$ unit normalization prior to quantizer training, encoding, and query table computation. For unit-normalized vectors ($\|u\| = \|v\| = 1.0$), squared Euclidean distance $\|u-v\|^2 = 2 - 2(u \cdot v)$ is strictly monotonic with cosine similarity. Aligning the metrics increased $M=25$ recall@10 from 0.61 (unnormalized metric mismatch) to **0.87** at **1,076 median QPS** ($8\times$ compression). At $M=5$ ($40\times$ compression), recall@10 is **0.26** at **2,919 median QPS**.

### Inverted File with Product Quantization (IVFPQ): Implemented

`IVFPQIndex` combines coarse cluster pruning (IVF) with vector compression (PQ) to achieve high search throughput alongside low memory footprint.

- **Residual Product Quantization**: Rather than quantizing raw vectors, `IVFPQIndex` subtracts assigned coarse centroids $c$ from normalized vectors $v$ to compute residual vectors $r = v - c$. `ProductQuantizer` is trained on these residual vectors.
- **Dot-Product Residual Scoring**: Because subtracting centroids destroys unit-vector norm ($L_2 \neq 1.0$), Euclidean translation invariance does not apply to cosine distance. `IVFPQIndex` directly evaluates the exact algebraic expansion:
  $$\text{Query} \cdot \text{Vector} = \text{Query} \cdot (c + r) = (\text{Query} \cdot c) + (\text{Query} \cdot r)$$
  1. $\text{Query} \cdot c$: Coarse centroid dot product, captured during coarse cluster routing.
  2. $\text{Query} \cdot r$: Sub-vector residual dot product, evaluated via fast pre-computed dot-product lookup tables (`computeDotProductTable` and `computeAsymmetricDotProduct`).
- **Empirical Evaluation (GloVe-50, 100k vectors, 500 held-out queries, $N_{list}=316$)**:
  - At $M=25$ ($8\times$ compression, 25 bytes/vec) and $N_{probe}=8$: achieves **14,588 median QPS** (28.3x brute force) with **0.7712 recall@10**.
  - At $M=25$ and $N_{probe}=32$: reaches **0.8536 recall@10** at **5,226 median QPS** (10.1x brute force), matching PQ alone's recall while searching via IVF inverted lists.
  - At $M=10$ ($20\times$ compression, 10 bytes/vec) and $N_{probe}=8$: reaches **15,941 median QPS** (30.9x brute force) with **0.4910 recall@10**.

### Hierarchical Navigable Small World (HNSW): Paper Study Complete, Implementation Planned

`HNSWIndex` implements a multi-layer graph index balancing high recall with sub-millisecond query latency.

#### 1. Metric Convention Decision: Distance vs. Similarity
The HNSW paper writes all algorithms and heap operations assuming **distance** (smaller is better). To prevent subtle comparison-inversion bugs while preserving exact equivalence with our unit-normalized cosine ground truth, all internal HNSW calculations define:
$$\text{distance}(u, v) = 1.0F - (u \cdot v)$$
For unit vectors, this distance is bounded in $[0.0, 2.0]$ and is strictly monotonic with cosine similarity. This preserves the paper's min-heap and max-heap logic intact without requiring inverted comparison operators.

#### 2. Structural Mechanics & Skip-List Hierarchy
- **Layers**: Top layers isolate long-range highway edges to skip across large distances with roughly logarithmic scaling on typical data. Layer 0 contains all vectors with short-range dense connections for precision.
- **Level Assignment**: Each vector's maximum level $l$ is sampled via $l = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor$, with $m_L = \frac{1}{\ln(M)}$. This is the skip-list default (promotion probability $p = 1/M$), presented in the paper as a reasonable default rather than a proven optimum — $m_L$ is a separate tunable knob.
- **Entry Point ($ep$)**: A persistent member variable pointing to the node at the topmost level ($L_{\text{max}}$). If a newly inserted node rolls level $l > L_{\text{max}}$, it becomes the new global entry point.
- **Search Traversal**:
  - Upper layers ($L_{\text{max}}$ down to 1): greedy single-best-hop routing ($ef = 1$) for maximum traversal speed.
  - Layer 0: bounded priority-queue beam search tracked by $efSearch \ge K$.
- **Heuristic Neighbor Selection**: Prunes candidate lists by enforcing directional diversity (only adding a neighbor if it is closer to the base node than to any already-selected neighbor). The paper selects **M** neighbors at every layer; $M_{\text{max}0} = 2M$ is only the shrink trigger cap, not the selection limit. The `keepPrunedConnections` flag is an optional experimental knob, not a default.
- **Edge Shrinking**: When a bidirectional link overflows a neighbor's connection list beyond $M_{\text{max}}$ (or $M_{\text{max}0}$), the neighbor's edge list is re-pruned using the heuristic selection rule, never by simple truncation. Note: shrinking can leave nodes unreachable (no remaining inbound edges). Full reachability from the entry point is not guaranteed by the algorithm.

#### 3. Heap Direction — Known Pitfall
`searchLayer` returns a max-heap on distance. `top()` yields the **furthest** candidate, not the nearest. This affects:
- `search()`: must drain the full heap and extract the k nearest from the sorted result, not pop k from the top.
- `add()`: must carry the nearest candidate (not the furthest) as the entry point for the next layer down. The paper passes the whole result set W as entry points.

Pseudocode that calls `candidates.top().id` to get the "best" entry is a bug — confirmed with a small repro. Implementation must follow the paper's Algorithms 1, 2, and 4 directly.

#### 4. Analytical Memory Model (GloVe-50, $M=16$)
- Vector coordinates: $50 \times 4\text{ B} = \mathbf{200\text{ B}}$
- Layer 0 links ($M_{\text{max}0} = 2M = 32$, `uint32_t` IDs): $32 \times 4\text{ B} = \mathbf{128\text{ B}}$
- Upper-layer links: Geometric series $\sum_{l=1}^{\infty} 16^{-l} = \frac{1}{15} \approx 6.7\%$ of Layer 0 links $\approx \mathbf{8.5\text{ B}}$
- **Analytical payload total**: $\approx \mathbf{336.5\text{ B / vector}}$ (compared to 200 B raw and 25 B PQ payload).
- *Allocator overhead note*: A dynamic `std::vector<std::vector<uint32_t>>` structure introduces allocator control blocks and heap fragmentation. Physical Resident Set Size (RSS) will be measured and compared against the analytical figure. The real HNSW vs IVFPQ memory ratio is approximately 5x (not 15-30x as raw payload alone might suggest, because IVFPQ's `std::vector<uint8_t>` per candidate carries ~60+ B of allocator overhead).

#### 5. Implementation Guards
- `ef >= k` must be enforced with `throw std::invalid_argument`, not `assert` (compiled out under `-DNDEBUG`).
- `search()` must normalize the query and reject zero-norm inputs, matching the contract of all other indexes.
- `selectNeighborsHeuristic` input must be sorted ascending by distance (drain and sort the heap first).
- A per-query distance-evaluation counter will be added to the benchmark to measure actual computational cost.

#### 6. Baseline & Performance Targets
- **Recall Target**: IVFPQ tops out at $0.85$ recall@10 on GloVe-50. IVF reaches $0.97$ recall at $718$ QPS ($N_{list}=316, N_{probe}=32$).
- **HNSW Target**: Reach $\ge 0.95$ recall@10 at $> 1,000$ QPS across $efSearch$ sweeps, surpassing the speed/recall ceiling of both IVF and IVFPQ.

## Storage Learning Notes

LSM trees favor write-heavy workloads through append-only segments and compaction. B+ trees favor point lookups and range scans through balanced in-place structure. The WAL exercise demonstrates that replaying complete append-only records can restore state after an interrupted write.

