#include "brute_force_index.hpp"
#include "glove_loader.hpp"
#include "ivf_index.hpp"

#include <algorithm>
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

struct QpsMeasurement {
    std::vector<double> samples;
    double median_qps;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
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
    return {samples, median(samples)};
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc < 2 || argc > 8) {
            throw std::invalid_argument(
                "Usage: glove_benchmark <glove_file> [dimensions] [index_count] [query_count] "
                "[top_k] [nlist] [repetitions]");
        }

        const std::string glove_path(argv[1]);
        const std::size_t dimensions = argc > 2 ? parsePositiveSize(argv[2], "dimensions") : 50;
        const std::size_t index_count = argc > 3 ? parsePositiveSize(argv[3], "index_count") : 100000;
        const std::size_t query_count = argc > 4 ? parsePositiveSize(argv[4], "query_count") : 500;
        const std::size_t top_k = argc > 5 ? parsePositiveSize(argv[5], "top_k") : 10;
        const std::size_t nlist = argc > 6 ? parsePositiveSize(argv[6], "nlist") : 316;
        const std::size_t repetitions = argc > 7 ? parsePositiveSize(argv[7], "repetitions") : 3;
        if (nlist > index_count) {
            throw std::invalid_argument("nlist cannot exceed index_count.");
        }

        const std::size_t query_pool_size = std::max(query_count * 10, static_cast<std::size_t>(5000));
        const vectordb::GloveEmbeddings embeddings =
            vectordb::loadGloveEmbeddings(glove_path, dimensions, index_count + query_pool_size);
        if (embeddings.size() <= index_count) {
            throw std::runtime_error("The GloVe file does not contain enough embeddings for the requested split.");
        }

        const std::size_t available_queries = embeddings.size() - index_count;
        const std::size_t actual_query_count = std::min(query_count, available_queries);

        const std::size_t index_value_count = index_count * dimensions;
        const std::vector<float> index_vectors(
            embeddings.vectors.begin(),
            embeddings.vectors.begin() + static_cast<std::ptrdiff_t>(index_value_count));

        // Sample query indices uniformly at random from held-out embeddings to eliminate frequency bias
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

        vectordb::BruteForceIndex brute_force_index(dimensions);
        for (std::size_t vector_id = 0; vector_id < index_count; ++vector_id) {
            const auto start = index_vectors.begin() + static_cast<std::ptrdiff_t>(vector_id * dimensions);
            brute_force_index.add(std::vector<float>(start, start + static_cast<std::ptrdiff_t>(dimensions)));
        }

        const auto build_start = std::chrono::steady_clock::now();
        vectordb::IVFIndex ivf_index(dimensions, nlist);
        ivf_index.build(index_vectors);
        const std::chrono::duration<double> build_elapsed = std::chrono::steady_clock::now() - build_start;
        const vectordb::IVFClusterStats cluster_stats = ivf_index.getClusterStats();

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
                  << "GloVe IVF cosine benchmark\n"
                  << "file: " << glove_path << "\n"
                  << "index vectors: " << index_count << ", random held-out queries: " << actual_query_count
                  << ", dimensions: " << dimensions << ", top_k: " << top_k << ", nlist: " << nlist
                  << ", repetitions: " << repetitions << '\n'
                  << "IVF build time: " << build_elapsed.count() << " s\n"
                  << "Cluster sizes -> min: " << cluster_stats.min_size
                  << ", max: " << cluster_stats.max_size
                  << ", mean: " << cluster_stats.mean_size
                  << ", stddev: " << cluster_stats.stddev_size
                  << ", empty: " << cluster_stats.empty_clusters << '\n'
                  << "Brute-force QPS samples: ";
        for (const double qps : brute_force_measurement.samples) {
            std::cout << qps << " ";
        }
        std::cout << "\nBrute-force median QPS: " << brute_force_measurement.median_qps << "\n\n"
                  << "nprobe";
        for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
            std::cout << "  run " << repetition + 1 << " QPS";
        }
        std::cout << "  median QPS  recall@" << top_k << '\n';

        constexpr std::array<std::size_t, 8> probe_values{1, 4, 8, 10, 12, 14, 16, 32};
        for (const std::size_t nprobe : probe_values) {
            if (nprobe > nlist) {
                continue;
            }

            std::vector<std::vector<vectordb::SearchResult>> approximate_results;
            approximate_results.reserve(actual_query_count);
            for (const std::vector<float>& query : queries) {
                approximate_results.push_back(ivf_index.search(query, top_k, nprobe));
            }

            double total_recall = 0.0;
            for (std::size_t query_id = 0; query_id < actual_query_count; ++query_id) {
                total_recall += recallAtK(exact_results[query_id], approximate_results[query_id]);
            }
            const double average_recall = total_recall / static_cast<double>(actual_query_count);
            const QpsMeasurement ivf_measurement = measureQps(
                actual_query_count,
                repetitions,
                [&ivf_index, &queries, top_k, nprobe](const std::size_t query_id) {
                    return ivf_index.search(queries[query_id], top_k, nprobe);
                });

            std::cout << std::setw(6) << nprobe;
            for (const double qps : ivf_measurement.samples) {
                std::cout << "  " << std::setw(9) << qps;
            }
            std::cout << "  " << std::setw(10) << ivf_measurement.median_qps
                      << "  " << average_recall << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
