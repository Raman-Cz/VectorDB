#include "hnsw_index.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace vectordb {
namespace {

float dotProduct(const float* left, const float* right, const std::size_t dimensions) {
    float score = 0.0F;
    for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
        score += left[dimension] * right[dimension];
    }
    return score;
}

bool normalize(float* vector, const std::size_t dimensions) {
    float squared_norm = 0.0F;
    for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
        squared_norm += vector[dimension] * vector[dimension];
    }

    if (squared_norm == 0.0F) {
        return false;
    }

    const float inverse_norm = 1.0F / std::sqrt(squared_norm);
    for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
        vector[dimension] *= inverse_norm;
    }
    return true;
}

// Nearer first; equal distances go to the lower ID so every ordering is deterministic.
template <typename CandidateType>
bool isNearer(const CandidateType& left, const CandidateType& right) {
    return left.distance < right.distance || (left.distance == right.distance && left.id < right.id);
}

}  // namespace

struct HNSWIndex::NearestOnTop {
    bool operator()(const Candidate& left, const Candidate& right) const {
        return isNearer(right, left);
    }
};

struct HNSWIndex::FurthestOnTop {
    bool operator()(const Candidate& left, const Candidate& right) const {
        return isNearer(left, right);
    }
};

HNSWIndex::HNSWIndex(const std::size_t dimensions, const HNSWConfig config)
    : dimensions_(dimensions), config_(config), level_multiplier_(0.0), generator_(config.seed) {
    if (dimensions_ == 0) {
        throw std::invalid_argument("Dimensions must be greater than zero.");
    }
    if (config_.M < 2) {
        throw std::invalid_argument("M must be at least 2.");
    }
    if (config_.ef_construction == 0) {
        throw std::invalid_argument("ef_construction must be greater than zero.");
    }

    // Skip-list default from the paper: roughly 1 in M nodes is promoted to each higher layer.
    level_multiplier_ = 1.0 / std::log(static_cast<double>(config_.M));
}

std::size_t HNSWIndex::add(const std::vector<float>& vector) {
    if (vector.size() != dimensions_) {
        throw std::invalid_argument("Vector dimensions do not match the index.");
    }
    if (size() >= std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("HNSWIndex node IDs are limited to uint32_t.");
    }

    std::vector<float> normalized = vector;
    if (!normalize(normalized.data(), dimensions_)) {
        throw std::invalid_argument("Vectors must have a non-zero norm.");
    }
    // Searches below read from this local copy because appending to vectors_ may reallocate it.
    const float* query = normalized.data();

    // uniform_real_distribution can return exactly 1.0 on some standard libraries (LWG 2524), which
    // would make log(0) infinite. Redrawing keeps 1 - u in (0, 1] so the level is always finite.
    std::uniform_real_distribution<double> unit_interval(0.0, 1.0);
    double uniform_draw = unit_interval(generator_);
    while (uniform_draw >= 1.0) {
        uniform_draw = unit_interval(generator_);
    }
    const int level = static_cast<int>(std::floor(-std::log(1.0 - uniform_draw) * level_multiplier_));

    const auto node = static_cast<std::uint32_t>(size());
    vectors_.insert(vectors_.end(), normalized.begin(), normalized.end());
    links_.emplace_back(static_cast<std::size_t>(level) + 1);

    if (max_level_ < 0) {
        entry_point_ = node;
        max_level_ = level;
        return node;
    }

    // Build cost is not reported, but searchLayer needs somewhere to count.
    std::size_t distance_computations = 0;
    std::vector<Candidate> entry_points{{distance(query, entry_point_), entry_point_}};

    // Phase 1: greedy descent (ef = 1) through layers that are above the new node's level.
    for (int current_level = max_level_; current_level > level; --current_level) {
        entry_points = searchLayer(query, entry_points, 1, current_level, distance_computations);
    }

    // Phase 2: on every layer the new node belongs to, find candidates and wire edges both ways.
    for (int current_level = std::min(level, max_level_); current_level >= 0; --current_level) {
        std::vector<Candidate> nearest = searchLayer(
            query, entry_points, config_.ef_construction, current_level, distance_computations);

        // The paper selects M neighbors on every layer; 2 * M on layer 0 is only the shrink cap.
        links_[node][current_level] = selectNeighborsHeuristic(nearest, config_.M);

        for (const std::uint32_t neighbor : links_[node][current_level]) {
            std::vector<std::uint32_t>& neighbor_links = links_[neighbor][current_level];
            neighbor_links.push_back(node);
            if (neighbor_links.size() <= maxLinks(current_level)) {
                continue;
            }

            // Shrink by re-running the heuristic around the neighbor, never by truncating.
            const float* neighbor_vector = vectors_.data() + (neighbor * dimensions_);
            std::vector<Candidate> neighbor_candidates;
            neighbor_candidates.reserve(neighbor_links.size());
            for (const std::uint32_t linked_id : neighbor_links) {
                neighbor_candidates.push_back({distance(neighbor_vector, linked_id), linked_id});
            }
            std::sort(neighbor_candidates.begin(), neighbor_candidates.end(), isNearer<Candidate>);
            neighbor_links = selectNeighborsHeuristic(neighbor_candidates, maxLinks(current_level));
        }

        // The whole candidate set W, not just its furthest top(), seeds the next layer down.
        entry_points = std::move(nearest);
    }

    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = node;
    }
    return node;
}

std::vector<SearchResult> HNSWIndex::search(
    const std::vector<float>& query,
    const std::size_t top_k,
    const std::size_t ef_search,
    HNSWSearchStats* stats) const {
    if (query.size() != dimensions_) {
        throw std::invalid_argument("Query dimensions do not match the index.");
    }
    if (top_k == 0) {
        return {};
    }
    if (ef_search < top_k) {
        throw std::invalid_argument("ef_search must be at least top_k.");
    }

    std::vector<float> normalized_query = query;
    if (!normalize(normalized_query.data(), dimensions_)) {
        throw std::invalid_argument("Queries must have a non-zero norm.");
    }
    if (max_level_ < 0) {
        return {};
    }

    const float* query_vector = normalized_query.data();
    std::size_t distance_computations = 1;
    std::vector<Candidate> entry_points{{distance(query_vector, entry_point_), entry_point_}};

    for (int level = max_level_; level > 0; --level) {
        entry_points = searchLayer(query_vector, entry_points, 1, level, distance_computations);
    }
    const std::vector<Candidate> nearest =
        searchLayer(query_vector, entry_points, ef_search, 0, distance_computations);

    // nearest is sorted nearest-first, so the first top_k entries are the answer.
    const std::size_t result_count = std::min(top_k, nearest.size());
    std::vector<SearchResult> results;
    results.reserve(result_count);
    for (std::size_t rank = 0; rank < result_count; ++rank) {
        results.push_back({nearest[rank].id, 1.0F - nearest[rank].distance});
    }

    if (stats != nullptr) {
        stats->distance_computations = distance_computations;
    }
    return results;
}

std::size_t HNSWIndex::countUnreachable() const {
    if (max_level_ < 0) {
        return 0;
    }

    std::vector<bool> visited(size(), false);
    std::vector<std::uint32_t> frontier{entry_point_};
    visited[entry_point_] = true;
    std::size_t reached = 1;
    while (!frontier.empty()) {
        const std::uint32_t node = frontier.back();
        frontier.pop_back();
        for (const std::uint32_t neighbor : links_[node][0]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                ++reached;
                frontier.push_back(neighbor);
            }
        }
    }
    return size() - reached;
}

std::size_t HNSWIndex::payloadBytes() const {
    std::size_t link_count = 0;
    for (const auto& node_links : links_) {
        for (const auto& level_links : node_links) {
            link_count += level_links.size();
        }
    }
    return vectors_.size() * sizeof(float) + link_count * sizeof(std::uint32_t);
}

std::size_t HNSWIndex::dimensions() const {
    return dimensions_;
}

std::size_t HNSWIndex::size() const {
    return vectors_.size() / dimensions_;
}

int HNSWIndex::maxLevel() const {
    return max_level_;
}

float HNSWIndex::distance(const float* query, const std::uint32_t id) const {
    // For unit vectors, 1 - cosine is in [0, 2] and orders exactly like cosine, reversed.
    return 1.0F - dotProduct(query, vectors_.data() + (id * dimensions_), dimensions_);
}

std::size_t HNSWIndex::maxLinks(const int level) const {
    return level == 0 ? 2 * config_.M : config_.M;
}

// Algorithm 2 of the paper: beam search of width ef on one layer.
// Returns up to ef candidates sorted nearest-first.
std::vector<HNSWIndex::Candidate> HNSWIndex::searchLayer(
    const float* query,
    const std::vector<Candidate>& entry_points,
    const std::size_t ef,
    const int level,
    std::size_t& distance_computations) const {
    std::vector<bool> visited(size(), false);
    // C in the paper: frontier still to expand, nearest on top.
    std::priority_queue<Candidate, std::vector<Candidate>, NearestOnTop> to_expand;
    // W in the paper: best ef found so far, furthest on top so it can be evicted cheaply.
    std::priority_queue<Candidate, std::vector<Candidate>, FurthestOnTop> nearest;

    for (const Candidate& entry : entry_points) {
        visited[entry.id] = true;
        to_expand.push(entry);
        nearest.push(entry);
        if (nearest.size() > ef) {
            nearest.pop();
        }
    }

    while (!to_expand.empty()) {
        const Candidate current = to_expand.top();
        // Every remaining frontier node is further than our worst kept result: nothing can improve.
        if (current.distance > nearest.top().distance) {
            break;
        }
        to_expand.pop();

        for (const std::uint32_t neighbor : links_[current.id][level]) {
            if (visited[neighbor]) {
                continue;
            }
            visited[neighbor] = true;

            const float neighbor_distance = distance(query, neighbor);
            ++distance_computations;
            if (nearest.size() < ef || neighbor_distance < nearest.top().distance) {
                to_expand.push({neighbor_distance, neighbor});
                nearest.push({neighbor_distance, neighbor});
                if (nearest.size() > ef) {
                    nearest.pop();
                }
            }
        }
    }

    // Draining a furthest-on-top heap yields worst-first, so sort before returning.
    std::vector<Candidate> sorted;
    sorted.reserve(nearest.size());
    while (!nearest.empty()) {
        sorted.push_back(nearest.top());
        nearest.pop();
    }
    std::sort(sorted.begin(), sorted.end(), isNearer<Candidate>);
    return sorted;
}

// Algorithm 4 of the paper, without extendCandidates or keepPrunedConnections.
// A candidate is kept only if it is closer to the base vector than to every neighbor already kept,
// which spreads links across directions instead of clustering them on one side.
std::vector<std::uint32_t> HNSWIndex::selectNeighborsHeuristic(
    const std::vector<Candidate>& sorted_candidates,
    const std::size_t max_neighbors) const {
    std::vector<std::uint32_t> selected;
    selected.reserve(max_neighbors);
    for (const Candidate& candidate : sorted_candidates) {
        if (selected.size() >= max_neighbors) {
            break;
        }

        const float* candidate_vector = vectors_.data() + (candidate.id * dimensions_);
        bool is_diverse = true;
        for (const std::uint32_t kept_id : selected) {
            if (distance(candidate_vector, kept_id) < candidate.distance) {
                is_diverse = false;
                break;
            }
        }
        if (is_diverse) {
            selected.push_back(candidate.id);
        }
    }
    return selected;
}

}
