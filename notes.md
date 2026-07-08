# Learning Log

## Day 1
Set up repo. Read pgvector architecture. Key insight: pgvector trades performance for PostgreSQL integration, making it simpler to deploy and operate.

## Phase 1 Complete — IR Foundations
Read Ch. 1, 2, 6, 7 of "Introduction to Information Retrieval" (Manning).

Key takeaway from Ch. 7:
- Cluster pruning = IVF in disguise. Leaders are centroids, followers are cluster members.
- IVF recall loss is a boundary/partition problem, not a local minimum problem — the query's true nearest neighbor may sit in an adjacent cluster that never gets examined.
- Approximate top-K is worth the trade-off: accept slightly-not-optimal for massive speed win.

## Phase 2 — Linear Algebra (next)
3Blue1Brown Essence of Linear Algebra Ch. 1-4, curse of dimensionality on Wikipedia.

## Phase 3 Start — Storage Engine Internals
Goal: DDIA Ch. 1-3 + Database Internals Ch. 2/6 → then build the append-only log toy.
Then: sync with repo scaffold, update DESIGN.md, write WAL demo.
