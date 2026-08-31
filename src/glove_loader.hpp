#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace vectordb {

struct GloveEmbeddings {
    std::vector<std::string> terms;
    std::vector<float> vectors;
    std::size_t dimensions;

    std::size_t size() const;
};

// Loads up to max_embeddings well-formed records from a standard GloVe text file.
GloveEmbeddings loadGloveEmbeddings(
    const std::string& path,
    std::size_t dimensions,
    std::size_t max_embeddings);

}
