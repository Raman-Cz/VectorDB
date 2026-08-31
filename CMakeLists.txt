cmake_minimum_required(VERSION 3.20)
project(VectorDB)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vector_db_core
    src/brute_force_index.cpp
    src/glove_loader.cpp
    src/ivf_index.cpp)
target_include_directories(vector_db_core PUBLIC src)

add_executable(vector_db src/main.cpp)
target_link_libraries(vector_db PRIVATE vector_db_core)

add_executable(ivf_benchmark src/ivf_benchmark.cpp)
target_link_libraries(ivf_benchmark PRIVATE vector_db_core)

add_executable(glove_benchmark src/glove_benchmark.cpp)
target_link_libraries(glove_benchmark PRIVATE vector_db_core)

enable_testing()
add_executable(brute_force_index_test tests/brute_force_index_test.cpp)
target_link_libraries(brute_force_index_test PRIVATE vector_db_core)
add_test(NAME brute_force_index_test COMMAND brute_force_index_test)

add_executable(ivf_index_test tests/ivf_index_test.cpp)
target_link_libraries(ivf_index_test PRIVATE vector_db_core)
add_test(NAME ivf_index_test COMMAND ivf_index_test)

add_executable(glove_loader_test tests/glove_loader_test.cpp)
target_link_libraries(glove_loader_test PRIVATE vector_db_core)
add_test(NAME glove_loader_test COMMAND glove_loader_test)
set_tests_properties(glove_loader_test PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
