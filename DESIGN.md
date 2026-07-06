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
