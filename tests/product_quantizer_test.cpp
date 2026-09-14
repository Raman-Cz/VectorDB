#include "product_quantizer.hpp"
#include "pq_index.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void testProductQuantizerBasic() {
    // 4D vectors split into M=2 sub-vectors of 2D each, 2 centroids
    vectordb::ProductQuantizer pq(4, 2, 2);
    assert(pq.dimensions() == 4);
    assert(pq.num_sub_vectors() == 2);
    assert(pq.sub_vector_dimensions() == 2);
    assert(pq.num_centroids() == 2);
    assert(!pq.isTrained());

    // Training dataset: 4 unit-normalized 4D vectors (L2 norm = 1.0)
    const float s = 1.0F / std::sqrt(2.0F); // ~0.7071
    const std::vector<float> dataset{
         s,  0.0F,  0.0F,  s,
         0.8F, 0.0F, 0.0F, 0.6F, // 0.8^2 + 0.6^2 = 1.0
        -s,  0.0F,  0.0F, -s,
        -0.8F, 0.0F, 0.0F, -0.6F
    };

    pq.train(dataset);
    assert(pq.isTrained());

    // Encode a unit-normalized query vector
    const std::vector<float> vec{s, 0.0F, 0.0F, s};
    const std::vector<std::uint8_t> codes = pq.encode(vec.data());
    assert(codes.size() == 2);

    // Decode vector back
    const std::vector<float> decoded = pq.decode(codes.data());
    assert(decoded.size() == 4);

    // Verify distance table calculation (2 sub-vectors x 2 centroids)
    const std::vector<float> dist_table = pq.computeDistanceTable(vec.data());
    assert(dist_table.size() == 4);

    const float adc_dist = pq.computeAsymmetricDistance(codes.data(), dist_table.data());
    assert(adc_dist >= 0.0F);
}

void testPQIndexMemoryAndSearch() {
    // 6D vectors split into M=3 sub-vectors of 2D each, 2 centroids
    vectordb::PQIndex index(6, 3, 2);
    assert(index.dimensions() == 6);
    assert(index.num_sub_vectors() == 3);
    assert(index.bytesPerVector() == 3);
    assert(!index.isBuilt());

    const std::vector<float> dataset{
        1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 0.0F,
        1.1F, 0.0F, 0.0F, 1.1F, 2.1F, 0.0F,
        -1.0F, 0.0F, 0.0F, -1.0F, -2.0F, 0.0F,
        -1.1F, 0.0F, 0.0F, -1.1F, -2.1F, 0.0F
    };

    index.trainAndBuild(dataset);
    assert(index.isBuilt());
    assert(index.size() == 4);
    assert(index.totalVectorDataBytes() == 4 * 3); // 4 vectors * 3 bytes

    const std::vector<float> query{1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 0.0F};
    const auto results = index.search(query, 2);
    assert(results.size() == 2);
    // Nearest result should be vector 0 or vector 1
    assert(results[0].id == 0 || results[0].id == 1);
}

void testValidation() {
    bool rejected_invalid_sub_vectors = false;
    try {
        vectordb::ProductQuantizer pq(5, 2); // 5 not divisible by 2
    } catch (const std::invalid_argument&) {
        rejected_invalid_sub_vectors = true;
    }
    assert(rejected_invalid_sub_vectors);
}

}  // namespace

int main() {
    testProductQuantizerBasic();
    testPQIndexMemoryAndSearch();
    testValidation();
    std::cout << "All ProductQuantizer unit tests passed successfully!\n";
    return 0;
}
