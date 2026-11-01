#pragma once

#include "search_result.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace vectordb {

struct HNSWConfig {
    // Neighbors selected per node per layer. Layer 0 may hold up to 2 * M links.
    std::size_t M = 16;
    // Beam width used while inserting. Larger values build a better graph more slowly.
    std::size_t ef_construction = 200;
    std::uint32_t seed = 42;
};

struct HNSWSearchStats {
    // Number of query-to-vector distance evaluations across all layers.
    std::size_t distance_computations = 0;
};

class HNSWIndex {
public:
    explicit HNSWIndex(std::size_t dimensions, HNSWConfig config = {});

    // Inserts one vector into the graph and returns its ID. IDs are assigned in insertion order.
    std::size_t add(const std::vector<float>& vector);

    // Greedy descent through the upper layers, then a beam search of width ef_search on layer 0.
    // ef_search must be at least top_k.
    std::vector<SearchResult> search(
        const std::vector<float>& query,
        std::size_t top_k,
        std::size_t ef_search = 50,
        HNSWSearchStats* stats = nullptr) const;

    // Counts nodes that cannot be reached from the entry point on layer 0.
    // Edge shrinking can drop a node's last inbound link, so this is reported, not prevented.
    std::size_t countUnreachable() const;

    // Graph payload: float coordinates plus uint32_t neighbor IDs, excluding allocator overhead.
    std::size_t payloadBytes() const;

    std::size_t dimensions() const;
    std::size_t size() const;
    int maxLevel() const;

private:
    struct Candidate {
        float distance;
        std::uint32_t id;
    };
    // Heap orderings. std::priority_queue keeps the "largest" element on top, so each name says
    // which candidate top() returns.
    struct NearestOnTop;
    struct FurthestOnTop;

    float distance(const float* query, std::uint32_t id) const;
    std::size_t maxLinks(int level) const;
    std::vector<Candidate> searchLayer(
        const float* query,
        const std::vector<Candidate>& entry_points,
        std::size_t ef,
        int level,
        std::size_t& distance_computations) const;
    std::vector<std::uint32_t> selectNeighborsHeuristic(
        const std::vector<Candidate>& sorted_candidates,
        std::size_t max_neighbors) const;

    std::size_t dimensions_;
    HNSWConfig config_;
    double level_multiplier_;
    std::mt19937 generator_;
    std::vector<float> vectors_;
    // links_[node][level] holds the neighbor IDs of node on that level.
    std::vector<std::vector<std::vector<std::uint32_t>>> links_;
    std::uint32_t entry_point_ = 0;
    int max_level_ = -1;
};

}
