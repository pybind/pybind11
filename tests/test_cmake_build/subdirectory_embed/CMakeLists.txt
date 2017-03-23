cmake_minimum_required(VERSION 3.0)
project(test_subdirectory_embed CXX)

add_subdirectory(${PYBIND11_PROJECT_DIR} pybind11)

add_executable(test_cmake_build ../embed.cpp)
target_link_libraries(test_cmake_build PRIVATE pybind11::embed)

add_custom_target(check $<TARGET_FILE:test_cmake_build> ${PROJECT_SOURCE_DIR}/../test.py)
