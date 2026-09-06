#pragma once

#include "search_result.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vectordb {

struct IVFBuildConfig {
    std::uint32_t seed = 42;
    std::size_t max_iterations = 100;
    float centroid_tolerance = 1e-4F;
};

struct IVFClusterStats {
    std::size_t min_size = 0;
    std::size_t max_size = 0;
    double mean_size = 0.0;
    double stddev_size = 0.0;
    std::size_t empty_clusters = 0;
};

class IVFIndex {
public:
    IVFIndex(std::size_t dimensions, std::size_t nlist, IVFBuildConfig config = {});

    // Builds the index once. The input is copied into the canonical contiguous store.
    void build(const std::vector<float>& vectors);

    // Searches the nearest nprobe inverted lists using normalized dot-product similarity.
    std::vector<SearchResult> search(
        const std::vector<float>& query,
        std::size_t top_k,
        std::size_t nprobe = 8) const;

    IVFClusterStats getClusterStats() const;

    std::size_t dimensions() const;
    std::size_t size() const;
    std::size_t nlist() const;
    bool isBuilt() const;

private:
    std::size_t dimensions_;
    std::size_t nlist_;
    IVFBuildConfig config_;
    std::vector<float> vectors_;
    std::vector<float> centroids_;
    std::vector<std::vector<std::size_t>> inverted_lists_;
};

}
