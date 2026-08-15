#pragma once

#include <cstddef>
#include <vector>

namespace vectordb {

struct SearchResult {
    std::size_t id;
    float score;
};

class BruteForceIndex {
public:
    explicit BruteForceIndex(std::size_t dimensions);

    std::size_t add(const std::vector<float>& vector);
    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t top_k) const;

    std::size_t dimensions() const;
    std::size_t size() const;

private:
    std::size_t dimensions_;
    std::vector<float> vectors_;
};

}
