#include "pq_index.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

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

// For Euclidean / ADC distance, LOWER score means closer neighbor (better result)
struct LowerDistanceScore {
    bool operator()(const SearchResult& left, const SearchResult& right) const {
        if (left.score == right.score) {
            return left.id < right.id;
        }
        return left.score < right.score;
    }
};

bool isBetterDistance(const SearchResult& left, const SearchResult& right) {
    return left.score < right.score || (left.score == right.score && left.id < right.id);
}

}  // namespace

PQIndex::PQIndex(
    const std::size_t dimensions,
    const std::size_t num_sub_vectors,
    const std::size_t num_centroids,
    const std::uint32_t seed)
    : dimensions_(dimensions),
      num_sub_vectors_(num_sub_vectors),
      quantizer_(dimensions, num_sub_vectors, num_centroids, seed) {}

void PQIndex::trainAndBuild(const std::vector<float>& vectors, const std::size_t max_iterations) {
    if (vectors.empty() || vectors.size() % dimensions_ != 0) {
        throw std::invalid_argument("Vectors must contain complete, non-empty records.");
    }

    vector_count_ = vectors.size() / dimensions_;
    std::vector<float> normalized_vectors = vectors;
    for (std::size_t i = 0; i < vector_count_; ++i) {
        normalize(normalized_vectors.data() + (i * dimensions_), dimensions_);
    }

    quantizer_.train(normalized_vectors, max_iterations);

    codes_.resize(vector_count_ * num_sub_vectors_);
    for (std::size_t i = 0; i < vector_count_; ++i) {
        const float* vec_ptr = normalized_vectors.data() + (i * dimensions_);
        const std::vector<std::uint8_t> code = quantizer_.encode(vec_ptr);
        std::copy(code.begin(), code.end(), codes_.data() + (i * num_sub_vectors_));
    }
}

std::vector<SearchResult> PQIndex::search(
    const std::vector<float>& query,
    const std::size_t top_k) const {
    if (!isBuilt()) {
        throw std::logic_error("The PQ index must be built before searching.");
    }
    if (query.size() != dimensions_) {
        throw std::invalid_argument("Query dimensions do not match the index.");
    }
    if (top_k == 0) {
        return {};
    }

    const std::size_t result_count = std::min(top_k, vector_count_);

    std::vector<float> normalized_query = query;
    normalize(normalized_query.data(), dimensions_);

    // Compute distance lookup table once per query (M x num_centroids)
    const std::vector<float> distance_table = quantizer_.computeDistanceTable(normalized_query.data());

    // Max-heap to track top_k smallest distances
    std::priority_queue<SearchResult, std::vector<SearchResult>, LowerDistanceScore> results;

    for (std::size_t i = 0; i < vector_count_; ++i) {
        const std::uint8_t* code_ptr = codes_.data() + (i * num_sub_vectors_);
        const float distance = quantizer_.computeAsymmetricDistance(code_ptr, distance_table.data());
        const SearchResult candidate{i, distance};

        if (results.size() < result_count) {
            results.push(candidate);
        } else if (isBetterDistance(candidate, results.top())) {
            results.pop();
            results.push(candidate);
        }
    }

    std::vector<SearchResult> sorted_results;
    sorted_results.reserve(results.size());
    while (!results.empty()) {
        sorted_results.push_back(results.top());
        results.pop();
    }
    std::sort(sorted_results.begin(), sorted_results.end(), isBetterDistance);
    return sorted_results;
}

std::size_t PQIndex::dimensions() const {
    return dimensions_;
}

std::size_t PQIndex::num_sub_vectors() const {
    return num_sub_vectors_;
}

std::size_t PQIndex::size() const {
    return vector_count_;
}

bool PQIndex::isBuilt() const {
    return quantizer_.isTrained() && vector_count_ > 0;
}

std::size_t PQIndex::bytesPerVector() const {
    return num_sub_vectors_;
}

std::size_t PQIndex::totalVectorDataBytes() const {
    return codes_.size();
}

}  // namespace vectordb
