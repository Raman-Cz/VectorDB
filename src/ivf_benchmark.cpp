#include "brute_force_index.hpp"
#include "ivf_index.hpp"

#include <array>
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

double recallAtK(
    const std::vector<vectordb::SearchResult>& exact_results,
    const std::vector<vectordb::SearchResult>& approximate_results) {
    if (exact_results.empty()) {
        return 1.0;
    }

    std::size_t matches = 0;
    for (const vectordb::SearchResult& approximate_result : approximate_results) {
        for (const vectordb::SearchResult& exact_result : exact_results) {
            if (approximate_result.id == exact_result.id) {
                ++matches;
                break;
            }
        }
    }
    return static_cast<double>(matches) / static_cast<double>(exact_results.size());
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const std::size_t vector_count = argc > 1 ? parsePositiveSize(argv[1], "vector_count") : 100000;
        const std::size_t dimensions = argc > 2 ? parsePositiveSize(argv[2], "dimensions") : 128;
        const std::size_t query_count = argc > 3 ? parsePositiveSize(argv[3], "query_count") : 100;
        const std::size_t top_k = argc > 4 ? parsePositiveSize(argv[4], "top_k") : 10;
        const std::size_t nlist = argc > 5 ? parsePositiveSize(argv[5], "nlist") : 316;
        if (argc > 6) {
            throw std::invalid_argument(
                "Usage: ivf_benchmark [vector_count] [dimensions] [query_count] [top_k] [nlist]");
        }
        if (nlist > vector_count) {
            throw std::invalid_argument("nlist cannot exceed vector_count.");
        }

        std::mt19937 generator(42);
        std::vector<float> dataset;
        dataset.reserve(vector_count * dimensions);
        vectordb::BruteForceIndex brute_force_index(dimensions);
        for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
            const std::vector<float> vector = randomVector(dimensions, generator);
            dataset.insert(dataset.end(), vector.begin(), vector.end());
            brute_force_index.add(vector);
        }

        std::vector<std::vector<float>> queries;
        queries.reserve(query_count);
        for (std::size_t query_id = 0; query_id < query_count; ++query_id) {
            queries.push_back(randomVector(dimensions, generator));
        }

        const auto build_start = std::chrono::steady_clock::now();
        vectordb::IVFIndex ivf_index(dimensions, nlist);
        ivf_index.build(dataset);
        const std::chrono::duration<double> build_elapsed = std::chrono::steady_clock::now() - build_start;

        std::vector<std::vector<vectordb::SearchResult>> exact_results;
        exact_results.reserve(query_count);
        const auto exact_start = std::chrono::steady_clock::now();
        for (const std::vector<float>& query : queries) {
            exact_results.push_back(brute_force_index.search(query, top_k));
        }
        const std::chrono::duration<double> exact_elapsed = std::chrono::steady_clock::now() - exact_start;

        const vectordb::IVFClusterStats cluster_stats = ivf_index.getClusterStats();

        std::cout << std::fixed << std::setprecision(2)
                  << "IVF cosine benchmark\n"
                  << "vectors: " << vector_count << ", dimensions: " << dimensions
                  << ", queries: " << query_count << ", top_k: " << top_k << ", nlist: " << nlist << '\n'
                  << "IVF build time: " << build_elapsed.count() << " s\n"
                  << "Cluster sizes -> min: " << cluster_stats.min_size
                  << ", max: " << cluster_stats.max_size
                  << ", mean: " << cluster_stats.mean_size
                  << ", stddev: " << cluster_stats.stddev_size
                  << ", empty: " << cluster_stats.empty_clusters << '\n'
                  << "Brute-force QPS: " << static_cast<double>(query_count) / exact_elapsed.count() << "\n\n"
                  << "nprobe  QPS       recall@" << top_k << '\n';

        constexpr std::array<std::size_t, 8> probe_values{1, 4, 8, 10, 12, 14, 16, 32};
        for (const std::size_t nprobe : probe_values) {
            if (nprobe > nlist) {
                continue;
            }

            std::vector<std::vector<vectordb::SearchResult>> approximate_results;
            approximate_results.reserve(query_count);
            const auto search_start = std::chrono::steady_clock::now();
            for (std::size_t query_id = 0; query_id < queries.size(); ++query_id) {
                approximate_results.push_back(ivf_index.search(queries[query_id], top_k, nprobe));
            }
            const std::chrono::duration<double> search_elapsed = std::chrono::steady_clock::now() - search_start;
            const double qps = static_cast<double>(query_count) / search_elapsed.count();

            double total_recall = 0.0;
            for (std::size_t query_id = 0; query_id < queries.size(); ++query_id) {
                total_recall += recallAtK(exact_results[query_id], approximate_results[query_id]);
            }
            const double average_recall = total_recall / static_cast<double>(query_count);

            std::cout << std::setw(6) << nprobe << "  "
                      << std::setw(8) << qps << "  "
                      << average_recall << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
