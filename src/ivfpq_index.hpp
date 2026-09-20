#pragma once

#include "ivf_index.hpp"
#include "product_quantizer.hpp"
#include "search_result.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vectordb {

struct IVFPQCandidate {
    std::size_t id;
    std::vector<std::uint8_t> codes;
};

class IVFPQIndex {
public:
    IVFPQIndex(
        std::size_t dimensions,
        std::size_t nlist,
        std::size_t num_sub_vectors,
        std::size_t num_centroids = 256,
        IVFBuildConfig config = {});

    // Trains coarse centroids, computes residuals, trains ProductQuantizer on residuals,
    // encodes residuals into M bytes per vector, and builds inverted lists.
    void build(const std::vector<float>& vectors, std::size_t max_pq_iterations = 50);

    // Searches nearest nprobe inverted lists using coarse dot product + sub-vector residual dot product lookups.
    std::vector<SearchResult> search(
        const std::vector<float>& query,
        std::size_t top_k,
        std::size_t nprobe = 8) const;

    IVFClusterStats getClusterStats() const;

    std::size_t dimensions() const;
    std::size_t size() const;
    std::size_t nlist() const;
    std::size_t num_sub_vectors() const;
    bool isBuilt() const;

    // Memory footprint reporting
    std::size_t bytesPerVector() const;
    std::size_t totalVectorDataBytes() const;

private:
    std::size_t dimensions_;
    std::size_t nlist_;
    std::size_t num_sub_vectors_;
    std::size_t num_centroids_;
    IVFBuildConfig config_;

    std::vector<float> coarse_centroids_;
    ProductQuantizer quantizer_;

    // Inverted lists holding candidates (vector ID + M byte codes)
    std::vector<std::vector<IVFPQCandidate>> inverted_lists_;
    std::size_t vector_count_ = 0;
};

}  // namespace vectordb
