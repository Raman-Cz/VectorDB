#include "glove_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vectordb {

std::size_t GloveEmbeddings::size() const {
    return vectors.size() / dimensions;
}

GloveEmbeddings loadGloveEmbeddings(
    const std::string& path,
    const std::size_t dimensions,
    const std::size_t max_embeddings) {
    if (dimensions == 0 || max_embeddings == 0) {
        throw std::invalid_argument("dimensions and max_embeddings must be greater than zero.");
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open GloVe file: " + path);
    }

    GloveEmbeddings embeddings{{}, {}, dimensions};
    embeddings.terms.reserve(max_embeddings);
    embeddings.vectors.reserve(max_embeddings * dimensions);

    std::string line;
    std::size_t line_number = 0;
    while (embeddings.size() < max_embeddings && std::getline(input, line)) {
        ++line_number;
        std::istringstream record(line);
        std::string term;
        if (!(record >> term)) {
            continue;
        }

        std::vector<float> vector(dimensions);
        for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
            if (!(record >> vector[dimension])) {
                throw std::runtime_error(
                    "Malformed GloVe record at line " + std::to_string(line_number) + ".");
            }
        }

        std::string extra_value;
        if (record >> extra_value) {
            throw std::runtime_error(
                "GloVe record at line " + std::to_string(line_number) + " has unexpected extra values.");
        }

        embeddings.terms.push_back(std::move(term));
        embeddings.vectors.insert(embeddings.vectors.end(), vector.begin(), vector.end());
    }

    return embeddings;
}

}
