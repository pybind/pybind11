find_package(Catch 2.13.2)

if(CATCH_FOUND)
  message(STATUS "Building pure C++ tests (not depending on Python) using Catch v${CATCH_VERSION}")
else()
  message(STATUS "Catch not detected. Interpreter tests will be skipped. Install Catch headers"
                 " manually or use `cmake -DDOWNLOAD_CATCH=ON` to fetch them automatically.")
  return()
endif()

add_executable(smart_holder_poc_test smart_holder_poc_test.cpp)
pybind11_enable_warnings(smart_holder_poc_test)
target_link_libraries(smart_holder_poc_test PRIVATE pybind11::headers Catch2::Catch2)

add_custom_target(
  test_pure_cpp
  COMMAND "$<TARGET_FILE:smart_holder_poc_test>"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

add_dependencies(check test_pure_cpp)
