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
- The benchmark defaults to 100,000 random 128-dimensional vectors and reports QPS.

Next: record a baseline run on the default dataset, then study why KD-trees, Ball Trees, and LSH degrade or lose competitiveness for high-dimensional embeddings before implementing IVF.
