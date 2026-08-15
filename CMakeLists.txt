cmake_minimum_required(VERSION 3.20)
project(VectorDB)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vector_db_core src/brute_force_index.cpp)
target_include_directories(vector_db_core PUBLIC src)

add_executable(vector_db src/main.cpp)
target_link_libraries(vector_db PRIVATE vector_db_core)

enable_testing()
add_executable(brute_force_index_test tests/brute_force_index_test.cpp)
target_link_libraries(brute_force_index_test PRIVATE vector_db_core)
add_test(NAME brute_force_index_test COMMAND brute_force_index_test)
