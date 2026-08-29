#include "ivf_index.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <random>
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

struct HigherScore {
    bool operator()(const SearchResult& left, const SearchResult& right) const {
        if (left.score == right.score) {
            return left.id > right.id;
        }
        return left.score > right.score;
    }
};

bool isBetter(const SearchResult& left, const SearchResult& right) {
    return left.score > right.score || (left.score == right.score && left.id < right.id);
}

std::size_t nearestCentroid(
    const float* vector,
    const std::vector<float>& centroids,
    const std::size_t dimensions,
    const std::size_t nlist) {
    std::size_t nearest = 0;
    float best_score = dotProduct(vector, centroids.data(), dimensions);
    for (std::size_t centroid_index = 1; centroid_index < nlist; ++centroid_index) {
        const float score = dotProduct(
            vector,
            centroids.data() + (centroid_index * dimensions),
            dimensions);
        if (score > best_score) {
            best_score = score;
            nearest = centroid_index;
        }
    }
    return nearest;
}

}  // namespace

IVFIndex::IVFIndex(
    const std::size_t dimensions,
    const std::size_t nlist,
    const IVFBuildConfig config)
    : dimensions_(dimensions), nlist_(nlist), config_(config) {
    if (dimensions_ == 0) {
        throw std::invalid_argument("Dimensions must be greater than zero.");
    }
    if (nlist_ == 0) {
        throw std::invalid_argument("nlist must be greater than zero.");
    }
    if (config_.max_iterations == 0) {
        throw std::invalid_argument("max_iterations must be greater than zero.");
    }
    if (config_.centroid_tolerance <= 0.0F) {
        throw std::invalid_argument("centroid_tolerance must be greater than zero.");
    }
}

void IVFIndex::build(const std::vector<float>& vectors) {
    if (vectors.empty() || vectors.size() % dimensions_ != 0) {
        throw std::invalid_argument("Vectors must contain complete, non-empty records.");
    }

    vectors_ = vectors;
    const std::size_t vector_count = size();
    if (nlist_ > vector_count) {
        throw std::invalid_argument("nlist cannot exceed the number of vectors.");
    }

    for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
        float* vector = vectors_.data() + (vector_id * dimensions_);
        if (!normalize(vector, dimensions_)) {
            throw std::invalid_argument("Vectors must have a non-zero norm.");
        }
    }

    std::vector<std::size_t> initial_ids(vector_count);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    std::mt19937 generator(config_.seed);
    std::shuffle(initial_ids.begin(), initial_ids.end(), generator);

    centroids_.assign(nlist_ * dimensions_, 0.0F);
    for (std::size_t centroid_index = 0; centroid_index < nlist_; ++centroid_index) {
        const float* initial_vector = vectors_.data() + (initial_ids[centroid_index] * dimensions_);
        std::copy_n(initial_vector, dimensions_, centroids_.data() + (centroid_index * dimensions_));
    }

    for (std::size_t iteration = 0; iteration < config_.max_iterations; ++iteration) {
        std::vector<float> centroid_sums(nlist_ * dimensions_, 0.0F);
        std::vector<std::size_t> cluster_sizes(nlist_, 0);

        for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
            const float* vector = vectors_.data() + (vector_id * dimensions_);
            const std::size_t centroid_index = nearestCentroid(vector, centroids_, dimensions_, nlist_);
            ++cluster_sizes[centroid_index];

            float* centroid_sum = centroid_sums.data() + (centroid_index * dimensions_);
            for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
                centroid_sum[dimension] += vector[dimension];
            }
        }

        std::vector<float> updated_centroids = centroids_;
        float total_movement = 0.0F;
        for (std::size_t centroid_index = 0; centroid_index < nlist_; ++centroid_index) {
            if (cluster_sizes[centroid_index] == 0) {
                continue;
            }

            float* updated_centroid = updated_centroids.data() + (centroid_index * dimensions_);
            const float* centroid_sum = centroid_sums.data() + (centroid_index * dimensions_);
            std::copy_n(centroid_sum, dimensions_, updated_centroid);

            // Normalizing each mean keeps spherical k-means aligned with cosine similarity.
            if (!normalize(updated_centroid, dimensions_)) {
                std::copy_n(
                    centroids_.data() + (centroid_index * dimensions_),
                    dimensions_,
                    updated_centroid);
            }

            const float* previous_centroid = centroids_.data() + (centroid_index * dimensions_);
            float squared_movement = 0.0F;
            for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
                const float difference = updated_centroid[dimension] - previous_centroid[dimension];
                squared_movement += difference * difference;
            }
            total_movement += std::sqrt(squared_movement);
        }

        centroids_ = std::move(updated_centroids);
        const float mean_movement = total_movement / static_cast<float>(nlist_);
        if (mean_movement < config_.centroid_tolerance) {
            break;
        }
    }

    inverted_lists_.assign(nlist_, {});
    for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
        const float* vector = vectors_.data() + (vector_id * dimensions_);
        const std::size_t centroid_index = nearestCentroid(vector, centroids_, dimensions_, nlist_);
        inverted_lists_[centroid_index].push_back(vector_id);
    }
}

std::vector<SearchResult> IVFIndex::search(
    const std::vector<float>& query,
    const std::size_t top_k,
    const std::size_t nprobe) const {
    if (!isBuilt()) {
        throw std::logic_error("The IVF index must be built before searching.");
    }
    if (query.size() != dimensions_) {
        throw std::invalid_argument("Query dimensions do not match the index.");
    }
    if (top_k == 0) {
        return {};
    }
    if (nprobe == 0) {
        throw std::invalid_argument("nprobe must be greater than zero.");
    }

    std::vector<float> normalized_query = query;
    if (!normalize(normalized_query.data(), dimensions_)) {
        throw std::invalid_argument("Queries must have a non-zero norm.");
    }

    const std::size_t probe_count = std::min(nprobe, nlist_);
    std::priority_queue<SearchResult, std::vector<SearchResult>, HigherScore> nearest_centroids;
    for (std::size_t centroid_index = 0; centroid_index < nlist_; ++centroid_index) {
        const SearchResult candidate{
            centroid_index,
            dotProduct(
                normalized_query.data(),
                centroids_.data() + (centroid_index * dimensions_),
                dimensions_)};
        if (nearest_centroids.size() < probe_count) {
            nearest_centroids.push(candidate);
        } else if (isBetter(candidate, nearest_centroids.top())) {
            nearest_centroids.pop();
            nearest_centroids.push(candidate);
        }
    }

    const std::size_t result_count = std::min(top_k, size());
    std::priority_queue<SearchResult, std::vector<SearchResult>, HigherScore> results;
    while (!nearest_centroids.empty()) {
        const std::size_t centroid_index = nearest_centroids.top().id;
        nearest_centroids.pop();
        for (const std::size_t vector_id : inverted_lists_[centroid_index]) {
            const float* vector = vectors_.data() + (vector_id * dimensions_);
            const SearchResult candidate{
                vector_id,
                dotProduct(normalized_query.data(), vector, dimensions_)};
            if (results.size() < result_count) {
                results.push(candidate);
            } else if (isBetter(candidate, results.top())) {
                results.pop();
                results.push(candidate);
            }
        }
    }

    std::vector<SearchResult> sorted_results;
    sorted_results.reserve(results.size());
    while (!results.empty()) {
        sorted_results.push_back(results.top());
        results.pop();
    }
    std::sort(sorted_results.begin(), sorted_results.end(), isBetter);
    return sorted_results;
}

std::size_t IVFIndex::dimensions() const {
    return dimensions_;
}

std::size_t IVFIndex::size() const {
    return vectors_.size() / dimensions_;
}

std::size_t IVFIndex::nlist() const {
    return nlist_;
}

bool IVFIndex::isBuilt() const {
    return !vectors_.empty() && centroids_.size() == nlist_ * dimensions_ && inverted_lists_.size() == nlist_;
}

}
