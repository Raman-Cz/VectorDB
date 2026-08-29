# Learning Log

## Phase 0: Mental Model

Reviewed vector database architecture trade-offs, including pgvector's choice to prioritize PostgreSQL integration and operational simplicity over specialized vector-index performance.

## Phase 1: Information Retrieval Foundations

Read Chapters 1, 2, 6, and 7 of *Introduction to Information Retrieval*.

- Cluster pruning is the conceptual predecessor of IVF: leaders correspond to centroids and followers correspond to assigned vectors.
- IVF recall loss is a partition-boundary problem: a true neighbor can be in an unprobed cluster.
- Approximate top-K is worthwhile when the speed gain justifies a controlled recall loss.

The Python BM25 file remains a lightweight learning stub and is not part of the C++ implementation path.

## Phase 2: Linear Algebra

Completed the required foundation for vector similarity: dot products, L1 and L2 norms, cosine similarity, normalization, and the curse of dimensionality. The key implementation consequence is that normalized vectors allow cosine similarity to be calculated as a dot product.

## Phase 3: Storage Internals

Completed the append-only WAL learning exercise in Python, including record checksums, replay, and crash-recovery tests. It is retained as a reference exercise rather than a production storage component.

## Phase 4: ANN Algorithms - Started

Implemented the exact brute-force cosine baseline in C++.

- `BruteForceIndex` normalizes vectors at insertion and queries at search time.
- Search scans every vector and returns sorted exact top-K results using a bounded min-heap.
- A CTest case verifies ranking, normalization, and rejection of zero-norm vectors.
- Vectors are stored in one flat contiguous `std::vector<float>` rather than individually allocated rows.
- The benchmark defaults to 100,000 random 128-dimensional vectors and reports QPS.
- Recorded baseline on 2026-08-15: 100 queries, top-K 10, 0.57 seconds total, 175.63 QPS. The CTest suite passed before recording this result.

### Failed Approaches: Complete

Completed the KD-tree, Ball Tree, and LSH study before beginning IVF work.

- KD-trees and Ball Trees lose pruning power as dimensions increase because distances and spatial partitions stop separating candidates effectively.
- LSH uses probabilistic bucket collisions to reduce comparisons, but its memory and recall trade-offs have generally been superseded by graph-based approaches for modern embedding search.
- These limitations motivate IVF's cluster-based candidate selection and, later, HNSW's graph traversal.

### IVF Reading: FAISS Paper

Read and simplified the IVFADC explanation from the FAISS paper. The roadmap's reference to Section 3 was corrected after checking the paper's actual structure:

- Section 2, "Problem Statement," contains the relevant IVFADC explanation.
- Section 3 is GPU architecture and k-selection, so it is intentionally skipped for the current single-threaded CPU scope.
- Basic IVF clusters vectors with k-means, assigns each vector to a centroid's inverted list, selects nearby centroids at query time, and scans only those lists.
- The paper's `tau` parameter is the same idea as `nprobe`: more probed lists improve recall but cost more comparisons.
- For the current 100,000-vector dataset, `sqrt(N)` suggests an initial experiment around 316 coarse clusters.
- Residual storage and product quantization improve compression, not the basic IVF routing mechanism; they remain separate Phase 4 PQ work.

### IVF Prediction Before Implementation

The brute-force baseline is 175.63 QPS, or approximately 5.7 ms per query, for 100,000 vectors at 128 dimensions. A first IVF experiment that probes roughly 10% of balanced clusters should scan about 10% of the dataset after centroid routing. Prediction: a 5-10x QPS improvement with some recall loss. Record the observed QPS and recall against this prediction when IVF is implemented.

Next: scope the chosen basic CPU IVF design in `DESIGN.md`. No IVF code has been started.
