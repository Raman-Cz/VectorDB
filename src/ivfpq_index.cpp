#include "ivfpq_index.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>

namespace vectordb {
namespace {

void normalize(float* vector, const std::size_t dimensions) {
    float squared_norm = 0.0F;
    for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
        squared_norm += vector[dimension] * vector[dimension];
    }
    if (squared_norm == 0.0F) {
        return;
    }
    const float inverse_norm = 1.0F / std::sqrt(squared_norm);
    for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
        vector[dimension] *= inverse_norm;
    }
}

float dotProduct(const float* left, const float* right, const std::size_t dimensions) {
    float product = 0.0F;
    for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
        product += left[dimension] * right[dimension];
    }
    return product;
}

std::size_t nearestCentroid(
    const float* vector,
    const float* centroids,
    const std::size_t dimensions,
    const std::size_t nlist) {
    std::size_t nearest = 0;
    float max_similarity = dotProduct(vector, centroids, dimensions);
    for (std::size_t k = 1; k < nlist; ++k) {
        const float similarity = dotProduct(
            vector,
            centroids + (k * dimensions),
            dimensions);
        if (similarity > max_similarity) {
            max_similarity = similarity;
            nearest = k;
        }
    }
    return nearest;
}

// Higher similarity score means closer neighbor for dot product ranking
struct LowerSimilarityScore {
    bool operator()(const SearchResult& left, const SearchResult& right) const {
        return left.score > right.score;
    }
};

}  // namespace

IVFPQIndex::IVFPQIndex(
    const std::size_t dimensions,
    const std::size_t nlist,
    const std::size_t num_sub_vectors,
    const std::size_t num_centroids,
    const IVFBuildConfig config)
    : dimensions_(dimensions),
      nlist_(nlist),
      num_sub_vectors_(num_sub_vectors),
      num_centroids_(num_centroids),
      config_(config),
      quantizer_(dimensions, num_sub_vectors, num_centroids, config.seed) {
    if (dimensions_ == 0) {
        throw std::invalid_argument("Dimensions must be greater than zero.");
    }
    if (nlist_ == 0) {
        throw std::invalid_argument("nlist must be greater than zero.");
    }
    if (num_sub_vectors_ == 0 || dimensions_ % num_sub_vectors_ != 0) {
        throw std::invalid_argument("Dimensions must be divisible by num_sub_vectors.");
    }
    if (num_centroids_ == 0 || num_centroids_ > 256) {
        throw std::invalid_argument("num_centroids must be between 1 and 256 for 1-byte encoding.");
    }
}

void IVFPQIndex::build(const std::vector<float>& vectors, const std::size_t max_pq_iterations) {
    if (vectors.empty() || vectors.size() % dimensions_ != 0) {
        throw std::invalid_argument("Vectors must contain complete, non-empty records.");
    }

    vector_count_ = vectors.size() / dimensions_;
    if (nlist_ > vector_count_) {
        throw std::invalid_argument("nlist cannot exceed vector_count.");
    }
    if (num_centroids_ > vector_count_) {
        throw std::invalid_argument("num_centroids cannot exceed vector_count.");
    }

    // 1. Copy and unit-normalize all data vectors
    std::vector<float> normalized_vectors = vectors;
    for (std::size_t i = 0; i < vector_count_; ++i) {
        normalize(normalized_vectors.data() + (i * dimensions_), dimensions_);
    }

    // 2. Initialize coarse centroids using spherical k-means
    coarse_centroids_.assign(nlist_ * dimensions_, 0.0F);

    std::mt19937 generator(config_.seed);
    std::vector<std::size_t> initial_ids(vector_count_);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    std::shuffle(initial_ids.begin(), initial_ids.end(), generator);

    for (std::size_t k = 0; k < nlist_; ++k) {
        const float* init_vector = normalized_vectors.data() + (initial_ids[k] * dimensions_);
        float* centroid = coarse_centroids_.data() + (k * dimensions_);
        std::copy_n(init_vector, dimensions_, centroid);
        normalize(centroid, dimensions_);
    }

    // 3. Train coarse centroids with spherical k-means
    for (std::size_t iter = 0; iter < config_.max_iterations; ++iter) {
        std::vector<float> centroid_sums(nlist_ * dimensions_, 0.0F);
        std::vector<std::size_t> cluster_sizes(nlist_, 0);

        for (std::size_t i = 0; i < vector_count_; ++i) {
            const float* vector = normalized_vectors.data() + (i * dimensions_);
            const std::size_t nearest = nearestCentroid(
                vector, coarse_centroids_.data(), dimensions_, nlist_);

            ++cluster_sizes[nearest];
            float* sum_ptr = centroid_sums.data() + (nearest * dimensions_);
            for (std::size_t d = 0; d < dimensions_; ++d) {
                sum_ptr[d] += vector[d];
            }
        }

        float total_movement = 0.0F;
        for (std::size_t k = 0; k < nlist_; ++k) {
            if (cluster_sizes[k] == 0) {
                continue;
            }
            float* centroid = coarse_centroids_.data() + (k * dimensions_);
            const float* sum_ptr = centroid_sums.data() + (k * dimensions_);
            std::vector<float> previous_centroid(centroid, centroid + dimensions_);

            std::copy_n(sum_ptr, dimensions_, centroid);
            normalize(centroid, dimensions_);

            const float movement = 1.0F - dotProduct(previous_centroid.data(), centroid, dimensions_);
            total_movement += std::max(0.0F, movement);
        }

        const float mean_movement = total_movement / static_cast<float>(nlist_);
        if (mean_movement < config_.centroid_tolerance) {
            break;
        }
    }

    // 4. Assign vectors to coarse clusters & compute residuals
    std::vector<std::size_t> assignments(vector_count_);
    std::vector<float> residuals(vector_count_ * dimensions_);

    for (std::size_t i = 0; i < vector_count_; ++i) {
        const float* vec = normalized_vectors.data() + (i * dimensions_);
        const std::size_t c_id = nearestCentroid(vec, coarse_centroids_.data(), dimensions_, nlist_);
        assignments[i] = c_id;

        const float* centroid = coarse_centroids_.data() + (c_id * dimensions_);
        float* res_ptr = residuals.data() + (i * dimensions_);
        for (std::size_t d = 0; d < dimensions_; ++d) {
            res_ptr[d] = vec[d] - centroid[d];
        }
    }

    // 5. Train ProductQuantizer on residuals
    quantizer_.train(residuals, max_pq_iterations);

    // 6. Encode residuals and populate inverted lists
    inverted_lists_.assign(nlist_, {});
    for (std::size_t i = 0; i < vector_count_; ++i) {
        const float* res_ptr = residuals.data() + (i * dimensions_);
        std::vector<std::uint8_t> codes = quantizer_.encode(res_ptr);
        inverted_lists_[assignments[i]].push_back(IVFPQCandidate{i, std::move(codes)});
    }
}

std::vector<SearchResult> IVFPQIndex::search(
    const std::vector<float>& query,
    const std::size_t top_k,
    const std::size_t nprobe) const {
    if (!isBuilt()) {
        throw std::logic_error("IVFPQIndex must be built before performing search.");
    }
    if (query.size() != dimensions_) {
        throw std::invalid_argument("Query dimensions must match index dimensions.");
    }
    if (top_k == 0) {
        throw std::invalid_argument("top_k must be greater than zero.");
    }
    if (nprobe == 0) {
        throw std::invalid_argument("nprobe must be greater than zero.");
    }

    const std::size_t probe_count = std::min(nprobe, nlist_);
    const std::size_t result_count = std::min(top_k, vector_count_);

    // 1. Normalize query vector
    std::vector<float> normalized_query = query;
    normalize(normalized_query.data(), dimensions_);

    // 2. Compute dot products between query and all coarse centroids
    std::vector<std::pair<float, std::size_t>> centroid_scores(nlist_);
    for (std::size_t k = 0; k < nlist_; ++k) {
        const float dot = dotProduct(
            normalized_query.data(),
            coarse_centroids_.data() + (k * dimensions_),
            dimensions_);
        centroid_scores[k] = {dot, k};
    }

    // Identify top probe_count centroids with highest dot products
    std::partial_sort(
        centroid_scores.begin(),
        centroid_scores.begin() + probe_count,
        centroid_scores.end(),
        [](const auto& left, const auto& right) {
            return left.first > right.first;
        });

    // 3. Pre-compute query sub-vector dot product lookup table for residuals
    const std::vector<float> dot_table = quantizer_.computeDotProductTable(normalized_query.data());

    // 4. Min-heap to maintain top result_count candidates
    std::priority_queue<SearchResult, std::vector<SearchResult>, LowerSimilarityScore> candidates;

    for (std::size_t p = 0; p < probe_count; ++p) {
        const float coarse_dot = centroid_scores[p].first;
        const std::size_t cluster_id = centroid_scores[p].second;
        const auto& list = inverted_lists_[cluster_id];

        for (const auto& candidate : list) {
            const float residual_dot = quantizer_.computeAsymmetricDotProduct(
                candidate.codes.data(), dot_table.data());
            const float total_score = coarse_dot + residual_dot;

            if (candidates.size() < result_count) {
                candidates.push(SearchResult{candidate.id, total_score});
            } else if (total_score > candidates.top().score) {
                candidates.pop();
                candidates.push(SearchResult{candidate.id, total_score});
            }
        }
    }

    // 5. Extract results sorted by score descending
    std::vector<SearchResult> results(candidates.size());
    for (std::size_t i = candidates.size(); i > 0; --i) {
        results[i - 1] = candidates.top();
        candidates.pop();
    }
    return results;
}

IVFClusterStats IVFPQIndex::getClusterStats() const {
    if (!isBuilt()) {
        return IVFClusterStats{};
    }

    IVFClusterStats stats;
    stats.min_size = inverted_lists_[0].size();
    stats.max_size = inverted_lists_[0].size();
    std::size_t total_assigned = 0;

    for (const auto& list : inverted_lists_) {
        const std::size_t cluster_size = list.size();
        stats.min_size = std::min(stats.min_size, cluster_size);
        stats.max_size = std::max(stats.max_size, cluster_size);
        total_assigned += cluster_size;
        if (cluster_size == 0) {
            ++stats.empty_clusters;
        }
    }

    stats.mean_size = static_cast<double>(total_assigned) / static_cast<double>(nlist_);

    double variance_sum = 0.0;
    for (const auto& list : inverted_lists_) {
        const double diff = static_cast<double>(list.size()) - stats.mean_size;
        variance_sum += diff * diff;
    }
    stats.stddev_size = std::sqrt(variance_sum / static_cast<double>(nlist_));

    return stats;
}

std::size_t IVFPQIndex::dimensions() const {
    return dimensions_;
}

std::size_t IVFPQIndex::size() const {
    return vector_count_;
}

std::size_t IVFPQIndex::nlist() const {
    return nlist_;
}

std::size_t IVFPQIndex::num_sub_vectors() const {
    return num_sub_vectors_;
}

bool IVFPQIndex::isBuilt() const {
    return !coarse_centroids_.empty();
}

std::size_t IVFPQIndex::bytesPerVector() const {
    return num_sub_vectors_;
}

std::size_t IVFPQIndex::totalVectorDataBytes() const {
    return vector_count_ * num_sub_vectors_;
}

}  // namespace vectordb
