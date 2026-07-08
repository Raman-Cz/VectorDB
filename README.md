# Vector DB Engine Project

A high-performance vector database implementation in C++ with focus on:
- Memory-efficient storage
- ANN search algorithms (HNSW, IVFFlat)
- Cache-aware design
- WAL-based crash recovery

## Language Choice
- **Core implementation:** C++ (for performance and systems credibility)
- **Learning exercises:** Python (for rapid prototyping of BM25, WAL demos)

## Learning Phases

| Phase | Topic | Status |
|-------|-------|--------|
| 0 | Mental Model (Qdrant, Weaviate, pgvector architecture) | Done |
| 1 | IR Foundations (Manning Ch. 1, 2, 6, 7) | Done |
| 2 | Linear Algebra (dot product, normalization, curse of dim.) | Next |
| 3 | Storage Engine Internals (DDIA, Database Internals, WAL toy) | In Progress |
| 4 | ANN Algorithms (IVF, PQ, HNSW) | Planned |
| 5 | Read FAISS Source | Planned |
| 6 | Memory/Cache Performance Model | Planned |
| 7 | Build the Vector Database (6-8 weeks) | Planned |

## Project Structure
```
VectorDB/
├── src/
│   ├── main.cpp              # C++ entry point
│   └── python/
│       ├── bm25_index.py     # BM25 exercise (Phase 1)
│       ├── append_only_log.py # WAL exercise (Phase 3)
│       └── test_wal.py       # Crash recovery tests
├── tests/
├── DESIGN.md                 # Design trade-offs
├── notes.md                  # Learning log
└── README.md
```
