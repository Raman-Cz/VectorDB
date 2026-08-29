#pragma once

#include <cstddef>

namespace vectordb {

struct SearchResult {
    std::size_t id;
    float score;
};

}
