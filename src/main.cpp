#include "brute_force_index.hpp"

#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t parsePositiveSize(const char* value, const char* argument_name) {
    try {
        const std::string text(value);
        std::size_t parsed_length = 0;
        const std::size_t parsed = std::stoull(text, &parsed_length);
        if (parsed == 0 || text.front() == '-' || parsed_length != text.size()) {
            throw std::invalid_argument("zero");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(argument_name) + " must be a positive integer.");
    }
}

std::vector<float> randomVector(const std::size_t dimensions, std::mt19937& generator) {
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    std::vector<float> vector(dimensions);
    for (float& value : vector) {
        value = distribution(generator);
    }
    return vector;
}

}

int main(int argc, char* argv[]) {
    try {
        const std::size_t vector_count = argc > 1 ? parsePositiveSize(argv[1], "vector_count") : 100000;
        const std::size_t dimensions = argc > 2 ? parsePositiveSize(argv[2], "dimensions") : 128;
        const std::size_t query_count = argc > 3 ? parsePositiveSize(argv[3], "query_count") : 100;
        const std::size_t top_k = argc > 4 ? parsePositiveSize(argv[4], "top_k") : 10;
        if (argc > 5) {
            throw std::invalid_argument("Usage: vector_db [vector_count] [dimensions] [query_count] [top_k]");
        }

        std::mt19937 generator(42);
        vectordb::BruteForceIndex index(dimensions);
        for (std::size_t id = 0; id < vector_count; ++id) {
            index.add(randomVector(dimensions, generator));
        }

        std::vector<std::vector<float>> queries;
        queries.reserve(query_count);
        for (std::size_t query = 0; query < query_count; ++query) {
            queries.push_back(randomVector(dimensions, generator));
        }

        float score_checksum = 0.0F;
        const auto start = std::chrono::steady_clock::now();
        for (const std::vector<float>& query : queries) {
            const std::vector<vectordb::SearchResult> results = index.search(query, top_k);
            score_checksum += results.front().score;
        }
        const auto end = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = end - start;
        const double qps = static_cast<double>(query_count) / elapsed.count();
        std::cout << std::fixed << std::setprecision(2)
                  << "Brute-force cosine benchmark\n"
                  << "vectors: " << index.size() << ", dimensions: " << index.dimensions()
                  << ", queries: " << query_count << ", top_k: " << top_k << '\n'
                  << "elapsed: " << elapsed.count() << " s\n"
                  << "QPS: " << qps << '\n'
                  << "score checksum: " << score_checksum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
