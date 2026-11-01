#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include "brute_force_index.hpp"
#include "hnsw_index.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::vector<float>> randomVectors(
    const std::size_t count,
    const std::size_t dimensions,
    const std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    std::vector<std::vector<float>> vectors(count, std::vector<float>(dimensions));
    for (std::vector<float>& vector : vectors) {
        for (float& value : vector) {
            value = distribution(generator);
        }
    }
    return vectors;
}

template <typename Function>
bool throwsInvalidArgument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// Fails if the graph is wired incorrectly or results come back in the wrong order.
// With 8 connected vectors and ef_search = 8, the beam can hold every node, so HNSW must be exact.
void testSmallGraphMatchesBruteForce() {
    const std::vector<std::vector<float>> dataset{
        {3.0F, 0.0F}, {2.0F, 1.0F}, {0.0F, 3.0F}, {-1.0F, 2.0F},
        {-3.0F, 0.0F}, {-2.0F, -1.0F}, {0.0F, -3.0F}, {1.0F, -2.0F}};

    vectordb::BruteForceIndex brute_force_index(2);
    vectordb::HNSWIndex hnsw_index(2);
    for (const std::vector<float>& vector : dataset) {
        brute_force_index.add(vector);
        hnsw_index.add(vector);
    }
    assert(hnsw_index.size() == dataset.size());
    assert(hnsw_index.countUnreachable() == 0);

    const std::vector<std::vector<float>> queries{
        {5.0F, 1.0F}, {-2.0F, 4.0F}, {-4.0F, -2.0F}, {1.0F, -5.0F}};
    for (const std::vector<float>& query : queries) {
        const auto exact_results = brute_force_index.search(query, 3);
        const auto hnsw_results = hnsw_index.search(query, 3, 8);
        assert(exact_results.size() == hnsw_results.size());
        for (std::size_t rank = 0; rank < exact_results.size(); ++rank) {
            assert(exact_results[rank].id == hnsw_results[rank].id);
            assert(std::fabs(exact_results[rank].score - hnsw_results[rank].score) < 1e-5F);
        }
    }
}

// Regression guard for the heap-direction pitfall in DESIGN.md. If search() or add() used the
// furthest candidate (max-heap top) as the "best" one, a stored vector would not find itself first.
void testStoredVectorFindsItselfFirst() {
    const auto dataset = randomVectors(1000, 16, 7);
    vectordb::HNSWIndex index(16);
    for (const std::vector<float>& vector : dataset) {
        index.add(vector);
    }

    for (std::size_t vector_id = 0; vector_id < dataset.size(); vector_id += 10) {
        const auto results = index.search(dataset[vector_id], 10, 50);
        assert(results.size() == 10);
        assert(results.front().id == vector_id);
        assert(std::fabs(results.front().score - 1.0F) < 1e-5F);
        for (std::size_t rank = 1; rank < results.size(); ++rank) {
            assert(results[rank - 1].score >= results[rank].score);
        }
    }
}

// Fails if neighbor selection, edge shrinking, or the layer-0 beam search lose good neighbors.
void testRecallAgainstBruteForce() {
    const std::size_t dimensions = 16;
    const auto dataset = randomVectors(2000, dimensions, 11);
    const auto queries = randomVectors(100, dimensions, 12);

    vectordb::BruteForceIndex brute_force_index(dimensions);
    vectordb::HNSWIndex hnsw_index(dimensions);
    for (const std::vector<float>& vector : dataset) {
        brute_force_index.add(vector);
        hnsw_index.add(vector);
    }

    std::size_t matches = 0;
    for (const std::vector<float>& query : queries) {
        const auto exact_results = brute_force_index.search(query, 10);
        const auto hnsw_results = hnsw_index.search(query, 10, 100);
        for (const auto& hnsw_result : hnsw_results) {
            for (const auto& exact_result : exact_results) {
                if (hnsw_result.id == exact_result.id) {
                    ++matches;
                    break;
                }
            }
        }
    }
    const double recall = static_cast<double>(matches) / static_cast<double>(queries.size() * 10);
    assert(recall >= 0.95);
}

// Fails if searchLayer loses its early stop or degenerates toward a scan of every vector.
// Measured at ef_search = 32: ~550 distance evaluations per query with the early stop and
// ~1300 without it, out of 5000 vectors. The 1/5 bound sits between the two.
void testSearchVisitsSmallFractionOfVectors() {
    const auto dataset = randomVectors(5000, 16, 21);
    vectordb::HNSWIndex index(16);
    for (const std::vector<float>& vector : dataset) {
        index.add(vector);
    }

    const auto queries = randomVectors(20, 16, 22);
    std::size_t total_distance_computations = 0;
    for (const std::vector<float>& query : queries) {
        vectordb::HNSWSearchStats stats;
        const auto results = index.search(query, 10, 32, &stats);
        assert(results.size() == 10);
        assert(stats.distance_computations > 0);
        total_distance_computations += stats.distance_computations;
    }
    const std::size_t mean_distance_computations = total_distance_computations / queries.size();
    assert(mean_distance_computations < index.size() / 5);
}

// Fails if the level generator or tie-breaking is not reproducible from the seed.
void testSameSeedBuildsSameGraph() {
    const auto dataset = randomVectors(500, 8, 31);
    vectordb::HNSWIndex first(8);
    vectordb::HNSWIndex second(8);
    for (const std::vector<float>& vector : dataset) {
        first.add(vector);
        second.add(vector);
    }
    assert(first.maxLevel() == second.maxLevel());
    assert(first.payloadBytes() == second.payloadBytes());

    const std::vector<float> query = randomVectors(1, 8, 32).front();
    const auto first_results = first.search(query, 10, 40);
    const auto second_results = second.search(query, 10, 40);
    for (std::size_t rank = 0; rank < first_results.size(); ++rank) {
        assert(first_results[rank].id == second_results[rank].id);
    }
}

void testPayloadBytes() {
    vectordb::HNSWIndex index(4);
    index.add({1.0F, 0.0F, 0.0F, 0.0F});
    assert(index.payloadBytes() == 4 * sizeof(float));

    // A second vector adds its coordinates plus one link in each direction on layer 0.
    index.add({0.0F, 1.0F, 0.0F, 0.0F});
    assert(index.payloadBytes() >= 2 * 4 * sizeof(float) + 2 * sizeof(std::uint32_t));
}

void testValidation() {
    assert(throwsInvalidArgument([] { vectordb::HNSWIndex index(0); }));
    assert(throwsInvalidArgument([] { vectordb::HNSWIndex index(2, {1, 200, 42}); }));
    assert(throwsInvalidArgument([] { vectordb::HNSWIndex index(2, {16, 0, 42}); }));

    vectordb::HNSWIndex index(2);
    assert(index.search({1.0F, 0.0F}, 1).empty());
    assert(throwsInvalidArgument([&index] { index.add({1.0F}); }));
    assert(throwsInvalidArgument([&index] { index.add({0.0F, 0.0F}); }));
    assert(index.size() == 0);

    index.add({1.0F, 0.0F});
    index.add({0.0F, 1.0F});
    assert(index.search({1.0F, 0.0F}, 0).empty());
    assert(throwsInvalidArgument([&index] { index.search({1.0F}, 1); }));
    assert(throwsInvalidArgument([&index] { index.search({0.0F, 0.0F}, 1); }));
    // DESIGN.md guard: ef_search < top_k throws instead of relying on an assert.
    assert(throwsInvalidArgument([&index] { index.search({1.0F, 0.0F}, 10, 5); }));
}

}  // namespace

int main() {
    testValidation();
    testPayloadBytes();
    testSmallGraphMatchesBruteForce();
    testStoredVectorFindsItselfFirst();
    testSameSeedBuildsSameGraph();
    testRecallAgainstBruteForce();
    testSearchVisitsSmallFractionOfVectors();
    return 0;
}
