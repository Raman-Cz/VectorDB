#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vectordb {

class ProductQuantizer {
public:
    // Dimensions must be divisible by num_sub_vectors.
    // Default num_centroids is 256 (fitting in 1 uint8_t byte).
    ProductQuantizer(
        std::size_t dimensions,
        std::size_t num_sub_vectors,
        std::size_t num_centroids = 256,
        std::uint32_t seed = 42);

    // Trains the M sub-space codebooks using k-means clustering.
    void train(const std::vector<float>& vectors, std::size_t max_iterations = 50);

    // Encodes a single D-dimensional vector into M byte codes.
    std::vector<std::uint8_t> encode(const float* vector) const;

    // Decodes M byte codes back into an approximate D-dimensional vector.
    std::vector<float> decode(const std::uint8_t* codes) const;

    // Pre-computes a distance lookup table (size M x num_centroids) for Asymmetric Distance Computation (ADC).
    // Stores squared Euclidean distance from query sub-vectors to all centroids.
    std::vector<float> computeDistanceTable(const float* query) const;

    // Fast ADC lookup: sums pre-computed sub-space distances using M byte codes.
    float computeAsymmetricDistance(const std::uint8_t* codes, const float* distance_table) const;

    std::size_t dimensions() const;
    std::size_t num_sub_vectors() const;
    std::size_t sub_vector_dimensions() const;
    std::size_t num_centroids() const;
    bool isTrained() const;

private:
    std::size_t dimensions_;
    std::size_t num_sub_vectors_;
    std::size_t sub_vector_dimensions_;
    std::size_t num_centroids_;
    std::uint32_t seed_;

    // Flattened codebooks: size (num_sub_vectors * num_centroids * sub_vector_dimensions)
    std::vector<float> codebooks_;
};

}  // namespace vectordb
