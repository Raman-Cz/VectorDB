#include "brute_force_index.hpp"
#include "ivf_index.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void assertSameResults(
    const std::vector<vectordb::SearchResult>& expected,
    const std::vector<vectordb::SearchResult>& actual) {
    assert(expected.size() == actual.size());
    for (std::size_t result_index = 0; result_index < expected.size(); ++result_index) {
        assert(expected[result_index].id == actual[result_index].id);
        assert(std::fabs(expected[result_index].score - actual[result_index].score) < 1e-5F);
    }
}

void testFullProbeMatchesBruteForce() {
    const std::vector<float> dataset{
        3.0F, 0.0F,
        2.0F, 1.0F,
        0.0F, 3.0F,
        -1.0F, 2.0F,
        -3.0F, 0.0F,
        -2.0F, -1.0F,
        0.0F, -3.0F,
        1.0F, -2.0F};

    vectordb::BruteForceIndex brute_force_index(2);
    for (std::size_t offset = 0; offset < dataset.size(); offset += 2) {
        brute_force_index.add({dataset[offset], dataset[offset + 1]});
    }

    vectordb::IVFIndex ivf_index(2, 3);
    ivf_index.build(dataset);

    const std::vector<std::vector<float>> queries{
        {5.0F, 1.0F},
        {-2.0F, 4.0F},
        {-4.0F, -2.0F},
        {1.0F, -5.0F}};
    for (const std::vector<float>& query : queries) {
        const std::vector<vectordb::SearchResult> exact_results = brute_force_index.search(query, 3);
        const std::vector<vectordb::SearchResult> full_probe_results =
            ivf_index.search(query, 3, ivf_index.nlist());
        assertSameResults(exact_results, full_probe_results);
    }
}

void testValidation() {
    vectordb::IVFIndex index(2, 2);

    bool rejected_unbuilt_search = false;
    try {
        index.search({1.0F, 0.0F}, 1);
    } catch (const std::logic_error&) {
        rejected_unbuilt_search = true;
    }
    assert(rejected_unbuilt_search);

    bool rejected_zero_vector = false;
    try {
        index.build({1.0F, 0.0F, 0.0F, 0.0F});
    } catch (const std::invalid_argument&) {
        rejected_zero_vector = true;
    }
    assert(rejected_zero_vector);

    index.build({1.0F, 0.0F, 0.0F, 1.0F});
    bool rejected_zero_probe_count = false;
    try {
        index.search({1.0F, 0.0F}, 1, 0);
    } catch (const std::invalid_argument&) {
        rejected_zero_probe_count = true;
    }
    assert(rejected_zero_probe_count);
}

}  // namespace

int main() {
    testFullProbeMatchesBruteForce();
    testValidation();
    return 0;
}
