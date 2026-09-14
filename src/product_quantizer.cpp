#include "product_quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace vectordb {
namespace {

float squaredEuclideanDistance(
    const float* left,
    const float* right,
    const std::size_t sub_vector_dimensions) {
    float distance = 0.0F;
    for (std::size_t dimension = 0; dimension < sub_vector_dimensions; ++dimension) {
        const float diff = left[dimension] - right[dimension];
        distance += diff * diff;
    }
    return distance;
}

std::size_t nearestSubCentroid(
    const float* sub_vector,
    const float* sub_codebook,
    const std::size_t sub_vector_dimensions,
    const std::size_t num_centroids) {
    std::size_t nearest = 0;
    float min_distance = squaredEuclideanDistance(sub_vector, sub_codebook, sub_vector_dimensions);
    for (std::size_t k = 1; k < num_centroids; ++k) {
        const float distance = squaredEuclideanDistance(
            sub_vector,
            sub_codebook + (k * sub_vector_dimensions),
            sub_vector_dimensions);
        if (distance < min_distance) {
            min_distance = distance;
            nearest = k;
        }
    }
    return nearest;
}

}  // namespace

ProductQuantizer::ProductQuantizer(
    const std::size_t dimensions,
    const std::size_t num_sub_vectors,
    const std::size_t num_centroids,
    const std::uint32_t seed)
    : dimensions_(dimensions),
      num_sub_vectors_(num_sub_vectors),
      num_centroids_(num_centroids),
      seed_(seed) {
    if (dimensions_ == 0) {
        throw std::invalid_argument("Dimensions must be greater than zero.");
    }
    if (num_sub_vectors_ == 0 || dimensions_ % num_sub_vectors_ != 0) {
        throw std::invalid_argument("Dimensions must be divisible by num_sub_vectors.");
    }
    if (num_centroids_ == 0 || num_centroids_ > 256) {
        throw std::invalid_argument("num_centroids must be between 1 and 256 for 1-byte encoding.");
    }
    sub_vector_dimensions_ = dimensions_ / num_sub_vectors_;
}

void ProductQuantizer::train(const std::vector<float>& vectors, const std::size_t max_iterations) {
    if (vectors.empty() || vectors.size() % dimensions_ != 0) {
        throw std::invalid_argument("Vectors must contain complete, non-empty records.");
    }

    const std::size_t vector_count = vectors.size() / dimensions_;
    if (num_centroids_ > vector_count) {
        throw std::invalid_argument("num_centroids cannot exceed vector_count.");
    }

    codebooks_.assign(num_sub_vectors_ * num_centroids_ * sub_vector_dimensions_, 0.0F);

    std::mt19937 generator(seed_);

    // Train each sub-space codebook independently
    for (std::size_t m = 0; m < num_sub_vectors_; ++m) {
        float* sub_codebook = codebooks_.data() + (m * num_centroids_ * sub_vector_dimensions_);

        // Extract sub-vectors for sub-space m
        std::vector<float> sub_vectors(vector_count * sub_vector_dimensions_);
        for (std::size_t i = 0; i < vector_count; ++i) {
            const float* src = vectors.data() + (i * dimensions_) + (m * sub_vector_dimensions_);
            float* dst = sub_vectors.data() + (i * sub_vector_dimensions_);
            std::copy_n(src, sub_vector_dimensions_, dst);
        }

        // Initialize centroids by random selection
        std::vector<std::size_t> initial_ids(vector_count);
        std::iota(initial_ids.begin(), initial_ids.end(), 0);
        std::shuffle(initial_ids.begin(), initial_ids.end(), generator);

        for (std::size_t k = 0; k < num_centroids_; ++k) {
            const float* init_vector = sub_vectors.data() + (initial_ids[k] * sub_vector_dimensions_);
            std::copy_n(init_vector, sub_vector_dimensions_, sub_codebook + (k * sub_vector_dimensions_));
        }

        // Run K-Means on sub-space m
        for (std::size_t iter = 0; iter < max_iterations; ++iter) {
            std::vector<float> centroid_sums(num_centroids_ * sub_vector_dimensions_, 0.0F);
            std::vector<std::size_t> cluster_sizes(num_centroids_, 0);

            for (std::size_t i = 0; i < vector_count; ++i) {
                const float* sub_vec = sub_vectors.data() + (i * sub_vector_dimensions_);
                const std::size_t nearest = nearestSubCentroid(
                    sub_vec, sub_codebook, sub_vector_dimensions_, num_centroids_);

                ++cluster_sizes[nearest];
                float* sum_ptr = centroid_sums.data() + (nearest * sub_vector_dimensions_);
                for (std::size_t d = 0; d < sub_vector_dimensions_; ++d) {
                    sum_ptr[d] += sub_vec[d];
                }
            }

            float total_movement = 0.0F;
            for (std::size_t k = 0; k < num_centroids_; ++k) {
                if (cluster_sizes[k] == 0) {
                    continue;
                }
                float* centroid = sub_codebook + (k * sub_vector_dimensions_);
                const float* sum_ptr = centroid_sums.data() + (k * sub_vector_dimensions_);

                float movement = 0.0F;
                for (std::size_t d = 0; d < sub_vector_dimensions_; ++d) {
                    const float new_val = sum_ptr[d] / static_cast<float>(cluster_sizes[k]);
                    const float diff = new_val - centroid[d];
                    movement += diff * diff;
                    centroid[d] = new_val;
                }
                total_movement += std::sqrt(movement);
            }

            const float mean_movement = total_movement / static_cast<float>(num_centroids_);
            if (mean_movement < 1e-4F) {
                break;
            }
        }
    }
}

std::vector<std::uint8_t> ProductQuantizer::encode(const float* vector) const {
    if (!isTrained()) {
        throw std::logic_error("ProductQuantizer must be trained before encoding.");
    }

    std::vector<std::uint8_t> codes(num_sub_vectors_);
    for (std::size_t m = 0; m < num_sub_vectors_; ++m) {
        const float* sub_vector = vector + (m * sub_vector_dimensions_);
        const float* sub_codebook = codebooks_.data() + (m * num_centroids_ * sub_vector_dimensions_);
        codes[m] = static_cast<std::uint8_t>(
            nearestSubCentroid(sub_vector, sub_codebook, sub_vector_dimensions_, num_centroids_));
    }
    return codes;
}

std::vector<float> ProductQuantizer::decode(const std::uint8_t* codes) const {
    if (!isTrained()) {
        throw std::logic_error("ProductQuantizer must be trained before decoding.");
    }

    std::vector<float> vector(dimensions_);
    for (std::size_t m = 0; m < num_sub_vectors_; ++m) {
        const std::size_t k = codes[m];
        const float* sub_centroid = codebooks_.data() + (m * num_centroids_ * sub_vector_dimensions_) +
                                    (k * sub_vector_dimensions_);
        std::copy_n(sub_centroid, sub_vector_dimensions_, vector.data() + (m * sub_vector_dimensions_));
    }
    return vector;
}

std::vector<float> ProductQuantizer::computeDistanceTable(const float* query) const {
    if (!isTrained()) {
        throw std::logic_error("ProductQuantizer must be trained before computing distance tables.");
    }

    std::vector<float> distance_table(num_sub_vectors_ * num_centroids_);
    for (std::size_t m = 0; m < num_sub_vectors_; ++m) {
        const float* query_sub_vec = query + (m * sub_vector_dimensions_);
        const float* sub_codebook = codebooks_.data() + (m * num_centroids_ * sub_vector_dimensions_);
        float* table_row = distance_table.data() + (m * num_centroids_);

        for (std::size_t k = 0; k < num_centroids_; ++k) {
            table_row[k] = squaredEuclideanDistance(
                query_sub_vec,
                sub_codebook + (k * sub_vector_dimensions_),
                sub_vector_dimensions_);
        }
    }
    return distance_table;
}

float ProductQuantizer::computeAsymmetricDistance(
    const std::uint8_t* codes,
    const float* distance_table) const {
    float distance = 0.0F;
    for (std::size_t m = 0; m < num_sub_vectors_; ++m) {
        distance += distance_table[(m * num_centroids_) + codes[m]];
    }
    return distance;
}

std::size_t ProductQuantizer::dimensions() const {
    return dimensions_;
}

std::size_t ProductQuantizer::num_sub_vectors() const {
    return num_sub_vectors_;
}

std::size_t ProductQuantizer::sub_vector_dimensions() const {
    return sub_vector_dimensions_;
}

std::size_t ProductQuantizer::num_centroids() const {
    return num_centroids_;
}

bool ProductQuantizer::isTrained() const {
    return !codebooks_.empty();
}

}  // namespace vectordb
