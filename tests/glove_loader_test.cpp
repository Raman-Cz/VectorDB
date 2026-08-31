#include "glove_loader.hpp"

#include <cassert>
#include <stdexcept>

int main() {
    const vectordb::GloveEmbeddings embeddings =
        vectordb::loadGloveEmbeddings("tests/data/tiny_glove.txt", 2, 2);
    assert(embeddings.dimensions == 2);
    assert(embeddings.size() == 2);
    assert(embeddings.terms[0] == "cat");
    assert(embeddings.vectors[3] == 1.0F);

    bool rejected_bad_dimensions = false;
    try {
        vectordb::loadGloveEmbeddings("tests/data/tiny_glove.txt", 3, 2);
    } catch (const std::runtime_error&) {
        rejected_bad_dimensions = true;
    }
    assert(rejected_bad_dimensions);

    return 0;
}
