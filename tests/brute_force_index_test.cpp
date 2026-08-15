#include "brute_force_index.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

int main() {
    vectordb::BruteForceIndex index(2);
    assert(index.add({2.0F, 0.0F}) == 0);
    assert(index.add({0.0F, 2.0F}) == 1);
    assert(index.add({-2.0F, 0.0F}) == 2);

    const std::vector<vectordb::SearchResult> results = index.search({4.0F, 0.0F}, 2);
    assert(results.size() == 2);
    assert(results[0].id == 0);
    assert(results[0].score == 1.0F);
    assert(results[1].id == 1);
    assert(results[1].score == 0.0F);

    bool rejected_zero_vector = false;
    try {
        index.add({0.0F, 0.0F});
    } catch (const std::invalid_argument&) {
        rejected_zero_vector = true;
    }
    assert(rejected_zero_vector);

    return 0;
}
