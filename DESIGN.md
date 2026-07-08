# Design Trade-offs

## pgvector Trade-off
pgvector makes the trade-off of sacrificing some search performance and scalability to integrate seamlessly with PostgreSQL. This allows for:
- No separate infrastructure needed
- Transactional consistency across relational and vector data
- Simpler deployment and maintenance
- Native SQL integration for complex queries

## C++ Implementation Strategy
- Use C++ for core storage engine and search algorithms
- Implement memory-aware data structures (cache lines, struct packing)
- Use Python for learning exercises (BM25 index, WAL demo)

## ANN Philosophy (from IR Ch. 7)
Exact top-K is often not worth the cost — approximate is fine if it's close. This is the entire justification for approximate nearest neighbor search.

### Cluster Pruning → IVF
Cluster pruning from classical IR (leaders/followers) is IVF's direct conceptual ancestor:
- **Leaders** = cluster centroids
- **Followers** = vectors assigned to nearest centroid
- **Query time**: search only the nearest cluster(s) instead of the full dataset
- **b1/b2 parameters** = IVF's `nprobe` (how many clusters to search)

Why IVF misses the true nearest neighbor: it's a boundary/partition problem, not a local minimum problem. A query near a cluster boundary may have its true nearest neighbor in an adjacent cluster that never gets examined. The fix is probing more clusters (wider net), not searching harder within one cluster.

### LSM vs B+ Tree
- **LSM trees**: write-optimized, append-only segments with periodic compaction. Better for write-heavy workloads.
- **B+ trees**: read-optimized, in-place updates with balanced tree structure. Better for point lookups and range scans.

### Crash Recovery
WAL (Write-Ahead Log) is append-only by design — this prevents corruption because:
1. Each mutation is appended to the log before touching the primary data structure
2. On restart, replay the log to reconstruct state
3. Append-only means you never overwrite a committed record — at worst you lose the last incomplete write
