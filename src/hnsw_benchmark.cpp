#include "brute_force_index.hpp"
#include "glove_loader.hpp"
#include "hnsw_index.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
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

struct QpsMeasurement {
    std::vector<double> samples;
    double median_qps = 0.0;
    double min_qps = 0.0;
    double max_qps = 0.0;
};

template <typename SearchFunction>
QpsMeasurement measureQps(
    const std::size_t query_count,
    const std::size_t repetitions,
    SearchFunction&& search) {
    std::vector<double> samples;
    samples.reserve(repetitions);
    std::size_t result_count = 0;
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t query_id = 0; query_id < query_count; ++query_id) {
            result_count += search(query_id).size();
        }
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        samples.push_back(static_cast<double>(query_count) / elapsed.count());
    }

    if (result_count == 0) {
        throw std::logic_error("Search returned no results during benchmarking.");
    }

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double median_qps = sorted[sorted.size() / 2];
    return {samples, median_qps, sorted.front(), sorted.back()};
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc < 2 || argc > 9) {
            throw std::invalid_argument(
                "Usage: hnsw_benchmark <glove_file> [dimensions] [index_count] [query_count] "
                "[top_k] [repetitions] [M] [ef_construction]");
        }

        const std::string glove_path(argv[1]);
        const std::size_t dimensions = argc > 2 ? parsePositiveSize(argv[2], "dimensions") : 50;
        const std::size_t index_count = argc > 3 ? parsePositiveSize(argv[3], "index_count") : 100000;
        const std::size_t query_count = argc > 4 ? parsePositiveSize(argv[4], "query_count") : 500;
        const std::size_t top_k = argc > 5 ? parsePositiveSize(argv[5], "top_k") : 10;
        const std::size_t repetitions = argc > 6 ? parsePositiveSize(argv[6], "repetitions") : 3;

        // Without M and ef_construction, sweep the roadmap grid: M {8, 16, 32} x efC {100, 200, 400}.
        const std::vector<std::size_t> m_values =
            argc > 7 ? std::vector<std::size_t>{parsePositiveSize(argv[7], "M")}
                     : std::vector<std::size_t>{8, 16, 32};
        const std::vector<std::size_t> ef_construction_values =
            argc > 8 ? std::vector<std::size_t>{parsePositiveSize(argv[8], "ef_construction")}
                     : std::vector<std::size_t>{100, 200, 400};

        const std::size_t query_pool_size = std::max(query_count * 10, static_cast<std::size_t>(5000));
        const vectordb::GloveEmbeddings embeddings =
            vectordb::loadGloveEmbeddings(glove_path, dimensions, index_count + query_pool_size);
        if (embeddings.size() <= index_count) {
            throw std::runtime_error("The GloVe file does not contain enough embeddings for the requested split.");
        }

        const std::size_t available_queries = embeddings.size() - index_count;
        const std::size_t actual_query_count = std::min(query_count, available_queries);

        // Sample query indices uniformly at random from held-out embeddings, as in the other benchmarks.
        std::vector<std::size_t> query_indices(available_queries);
        for (std::size_t i = 0; i < available_queries; ++i) {
            query_indices[i] = index_count + i;
        }
        std::mt19937 generator(42);
        std::shuffle(query_indices.begin(), query_indices.end(), generator);
        query_indices.resize(actual_query_count);

        std::vector<std::vector<float>> queries;
        queries.reserve(actual_query_count);
        for (const std::size_t global_id : query_indices) {
            const auto start = embeddings.vectors.begin() + static_cast<std::ptrdiff_t>(global_id * dimensions);
            queries.emplace_back(start, start + static_cast<std::ptrdiff_t>(dimensions));
        }

        std::vector<std::vector<float>> index_vectors;
        index_vectors.reserve(index_count);
        for (std::size_t vector_id = 0; vector_id < index_count; ++vector_id) {
            const auto start = embeddings.vectors.begin() + static_cast<std::ptrdiff_t>(vector_id * dimensions);
            index_vectors.emplace_back(start, start + static_cast<std::ptrdiff_t>(dimensions));
        }

        vectordb::BruteForceIndex brute_force_index(dimensions);
        for (const std::vector<float>& vector : index_vectors) {
            brute_force_index.add(vector);
        }

        std::vector<std::vector<vectordb::SearchResult>> exact_results;
        exact_results.reserve(actual_query_count);
        for (const std::vector<float>& query : queries) {
            exact_results.push_back(brute_force_index.search(query, top_k));
        }

        const QpsMeasurement brute_force_measurement = measureQps(
            actual_query_count,
            repetitions,
            [&brute_force_index, &queries, top_k](const std::size_t query_id) {
                return brute_force_index.search(queries[query_id], top_k);
            });

        std::cout << std::fixed << std::setprecision(2)
                  << "HNSW (Hierarchical Navigable Small World) Benchmark\n"
                  << "file: " << glove_path << "\n"
                  << "index vectors: " << index_count << ", random held-out queries: " << actual_query_count
                  << ", dimensions: " << dimensions << ", top_k: " << top_k << '\n'
                  << "Raw vector size: " << dimensions * sizeof(float) << " bytes\n"
                  << "Brute-force median QPS: " << brute_force_measurement.median_qps << "\n\n";

        const std::vector<std::size_t> ef_search_values = {10, 20, 40, 80, 160, 320};

        for (const std::size_t M : m_values) {
            for (const std::size_t ef_construction : ef_construction_values) {
                const auto build_start = std::chrono::steady_clock::now();
                vectordb::HNSWIndex hnsw(dimensions, {M, ef_construction, 42});
                for (const std::vector<float>& vector : index_vectors) {
                    hnsw.add(vector);
                }
                const std::chrono::duration<double> build_elapsed = std::chrono::steady_clock::now() - build_start;

                const double bytes_per_vector =
                    static_cast<double>(hnsw.payloadBytes()) / static_cast<double>(hnsw.size());

                std::cout << "=========================================================================================\n"
                          << "M: " << M << ", ef_construction: " << ef_construction
                          << ", build: " << std::setprecision(2) << build_elapsed.count() << "s"
                          << ", top level: " << hnsw.maxLevel()
                          << ", unreachable: " << hnsw.countUnreachable()
                          << ", payload: " << bytes_per_vector << " B/vector\n"
                          << "-----------------------------------------------------------------------------------------\n"
                          << std::left << std::setw(12) << "ef_search"
                          << std::setw(14) << "Median QPS"
                          << std::setw(18) << "QPS Range"
                          << std::setw(14) << "Speedup"
                          << std::setw(16) << "Dist/Query"
                          << std::setw(12) << "Recall@10" << "\n";

                for (const std::size_t ef_search : ef_search_values) {
                    if (ef_search < top_k) {
                        continue;
                    }

                    double total_recall = 0.0;
                    std::size_t total_distance_computations = 0;
                    for (std::size_t query_id = 0; query_id < actual_query_count; ++query_id) {
                        vectordb::HNSWSearchStats stats;
                        const auto results = hnsw.search(queries[query_id], top_k, ef_search, &stats);
                        total_recall += recallAtK(exact_results[query_id], results);
                        total_distance_computations += stats.distance_computations;
                    }
                    const double mean_recall = total_recall / static_cast<double>(actual_query_count);
                    const double mean_distance_computations =
                        static_cast<double>(total_distance_computations) / static_cast<double>(actual_query_count);

                    const QpsMeasurement hnsw_measurement = measureQps(
                        actual_query_count,
                        repetitions,
                        [&hnsw, &queries, top_k, ef_search](const std::size_t query_id) {
                            return hnsw.search(queries[query_id], top_k, ef_search);
                        });

                    std::stringstream qps_range_ss;
                    qps_range_ss << std::fixed << std::setprecision(0)
                                 << hnsw_measurement.min_qps << "-" << hnsw_measurement.max_qps;

                    std::stringstream speedup_ss;
                    speedup_ss << std::fixed << std::setprecision(1)
                               << hnsw_measurement.median_qps / brute_force_measurement.median_qps << "x";

                    std::cout << std::left << std::setw(12) << ef_search
                              << std::setw(14) << std::fixed << std::setprecision(2) << hnsw_measurement.median_qps
                              << std::setw(18) << qps_range_ss.str()
                              << std::setw(14) << speedup_ss.str()
                              << std::setw(16) << std::setprecision(0) << mean_distance_computations
                              << std::setw(12) << std::setprecision(4) << mean_recall << "\n";
                }
            }
        }
        std::cout << "=========================================================================================\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
