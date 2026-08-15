#include "brute_force_index.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace vectordb {
namespace {

void normalize(std::vector<float>& vector) {
    float squared_norm = 0.0F;
    for (const float value : vector) {
        squared_norm += value * value;
    }

    if (squared_norm == 0.0F) {
        throw std::invalid_argument("Vectors must have a non-zero norm.");
    }

    const float inverse_norm = 1.0F / std::sqrt(squared_norm);
    for (float& value : vector) {
        value *= inverse_norm;
    }
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

}

BruteForceIndex::BruteForceIndex(const std::size_t dimensions) : dimensions_(dimensions) {
    if (dimensions_ == 0) {
        throw std::invalid_argument("Dimensions must be greater than zero.");
    }
}

std::size_t BruteForceIndex::add(const std::vector<float>& vector) {
    if (vector.size() != dimensions_) {
        throw std::invalid_argument("Vector dimensions do not match the index.");
    }

    std::vector<float> normalized_vector = vector;
    normalize(normalized_vector);

    const std::size_t id = size();
    vectors_.insert(vectors_.end(), normalized_vector.begin(), normalized_vector.end());
    return id;
}

std::vector<SearchResult> BruteForceIndex::search(
    const std::vector<float>& query,
    const std::size_t top_k) const {
    if (query.size() != dimensions_) {
        throw std::invalid_argument("Query dimensions do not match the index.");
    }
    if (top_k == 0 || vectors_.empty()) {
        return {};
    }

    std::vector<float> normalized_query = query;
    normalize(normalized_query);
    const std::size_t result_count = std::min(top_k, size());

    std::priority_queue<SearchResult, std::vector<SearchResult>, HigherScore> results;
    for (std::size_t id = 0; id < size(); ++id) {
        const float* candidate = vectors_.data() + (id * dimensions_);
        float score = 0.0F;
        for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
            score += normalized_query[dimension] * candidate[dimension];
        }

        const SearchResult result{id, score};
        if (results.size() < result_count) {
            results.push(result);
        } else if (isBetter(result, results.top())) {
            results.pop();
            results.push(result);
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

std::size_t BruteForceIndex::dimensions() const {
    return dimensions_;
}

std::size_t BruteForceIndex::size() const {
    return vectors_.size() / dimensions_;
}

}
