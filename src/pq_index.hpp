#pragma once

#include "product_quantizer.hpp"
#include "search_result.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vectordb {

class PQIndex {
public:
    PQIndex(
        std::size_t dimensions,
        std::size_t num_sub_vectors,
        std::size_t num_centroids = 256,
        std::uint32_t seed = 42);

    // Trains the quantizer on the input vectors and encodes them into contiguous byte codes.
    void trainAndBuild(const std::vector<float>& vectors, std::size_t max_iterations = 50);

    // Performs Asymmetric Distance Computation (ADC) search returning approximate top_k results.
    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t top_k) const;

    std::size_t dimensions() const;
    std::size_t num_sub_vectors() const;
    std::size_t size() const;
    bool isBuilt() const;

    // Memory footprint reporting
    std::size_t bytesPerVector() const;
    std::size_t totalVectorDataBytes() const;

private:
    std::size_t dimensions_;
    std::size_t num_sub_vectors_;
    ProductQuantizer quantizer_;

    // Flat byte codes: size (vector_count * num_sub_vectors)
    std::vector<std::uint8_t> codes_;
    std::size_t vector_count_ = 0;
};

}  // namespace vectordb
