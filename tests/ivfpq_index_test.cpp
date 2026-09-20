#include "brute_force_index.hpp"
#include "ivfpq_index.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void testIVFPQIndexBasic() {
    // 6D vectors, nlist = 2, M = 3 sub-vectors of 2D each, 2 centroids per sub-space
    vectordb::IVFPQIndex index(6, 2, 3, 2);
    assert(index.dimensions() == 6);
    assert(index.nlist() == 2);
    assert(index.num_sub_vectors() == 3);
    assert(index.bytesPerVector() == 3);
    assert(!index.isBuilt());

    const std::vector<float> dataset{
        1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 0.0F,
        1.1F, 0.0F, 0.0F, 1.1F, 2.1F, 0.0F,
        -1.0F, 0.0F, 0.0F, -1.0F, -2.0F, 0.0F,
        -1.1F, 0.0F, 0.0F, -1.1F, -2.1F, 0.0F
    };

    index.build(dataset);
    assert(index.isBuilt());
    assert(index.size() == 4);
    assert(index.totalVectorDataBytes() == 4 * 3); // 4 vectors * 3 bytes

    const auto stats = index.getClusterStats();
    assert(stats.min_size + stats.max_size > 0);
    assert(stats.empty_clusters == 0 || stats.empty_clusters == 1);

    const std::vector<float> query{1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 0.0F};
    const auto results = index.search(query, 2, 2); // Probe both clusters
    assert(results.size() == 2);
    // Nearest result should be vector 0 or 1
    assert(results[0].id == 0 || results[0].id == 1);
}

void testIVFPQFullProbingRecall() {
    // 4D unit vectors, nlist = 2, M = 2 sub-vectors of 2D each
    vectordb::IVFPQIndex index(4, 2, 2, 2);
    vectordb::BruteForceIndex brute(4);

    const float s = 1.0F / std::sqrt(2.0F);
    const std::vector<float> dataset{
         s,  0.0F,  0.0F,  s,
         0.8F, 0.0F, 0.0F, 0.6F,
        -s,  0.0F,  0.0F, -s,
        -0.8F, 0.0F, 0.0F, -0.6F
    };

    for (std::size_t i = 0; i < 4; ++i) {
        const float* vec_ptr = dataset.data() + (i * 4);
        brute.add(std::vector<float>(vec_ptr, vec_ptr + 4));
    }
    index.build(dataset);

    const std::vector<float> query{s, 0.0F, 0.0F, s};
    const auto brute_results = brute.search(query, 2);
    const auto ivfpq_results = index.search(query, 2, 2); // nprobe = nlist = 2

    assert(brute_results.size() == 2);
    assert(ivfpq_results.size() == 2);
    assert(ivfpq_results[0].id == brute_results[0].id);
}

void testValidation() {
    bool rejected_invalid_sub_vectors = false;
    try {
        vectordb::IVFPQIndex index(5, 2, 2); // 5 not divisible by 2
    } catch (const std::invalid_argument&) {
        rejected_invalid_sub_vectors = true;
    }
    assert(rejected_invalid_sub_vectors);

    bool rejected_unbuilt_search = false;
    try {
        vectordb::IVFPQIndex index(4, 2, 2);
        index.search({1.0F, 0.0F, 0.0F, 0.0F}, 1);
    } catch (const std::logic_error&) {
        rejected_unbuilt_search = true;
    }
    assert(rejected_unbuilt_search);

    bool rejected_zero_vector = false;
    try {
        vectordb::IVFPQIndex index(4, 2, 2);
        index.build({0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F});
    } catch (const std::invalid_argument&) {
        rejected_zero_vector = true;
    }
    assert(rejected_zero_vector);

    bool rejected_zero_query = false;
    try {
        vectordb::IVFPQIndex index(4, 2, 2);
        index.build({1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F});
        index.search({0.0F, 0.0F, 0.0F, 0.0F}, 1);
    } catch (const std::invalid_argument&) {
        rejected_zero_query = true;
    }
    assert(rejected_zero_query);
}

}  // namespace

int main() {
    testIVFPQIndexBasic();
    testIVFPQFullProbingRecall();
    testValidation();
    std::cout << "All IVFPQIndex unit tests passed successfully!\n";
    return 0;
}
