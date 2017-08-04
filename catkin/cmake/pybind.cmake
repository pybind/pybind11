cmake_minimum_required(VERSION 2.8.3)

message(STATUS "Compiling using c++ 11 (required by PyBind11)")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../tools")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

set(TEMP_PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
if(DEFINED PYBIND_PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE ${PYBIND_PYTHON_EXECUTABLE} CACHE INTERNAL "")
endif()

include(pybind11Tools)

# Cache variables so pybind11_add_module can be used in parent projects
set(PYBIND11_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" CACHE INTERNAL "")
set(PYTHON_LIBRARIES ${PYTHON_LIBRARIES} CACHE INTERNAL "")

set(PYTHON_EXECUTABLE ${TEMP_PYTHON_EXECUTABLE} CACHE INTERNAL "")

macro(pybind_add_module target_name other)
    pybind11_add_module(${ARGV})
    target_link_libraries(${target_name} PRIVATE ${PYTHON_LIBRARIES} ${catkin_LIBRARIES})
    set_target_properties(${target_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_GLOBAL_PYTHON_DESTINATION})
    set(PYTHON_LIB_DIR ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION})
    add_custom_command(TARGET ${target_name}
      POST_BUILD
      COMMAND mkdir -p ${PYTHON_LIB_DIR} && cp $<TARGET_FILE:${target_name}> ${PYTHON_LIB_DIR}/${target_name}.so
      WORKING_DIRECTORY ${CATKIN_DEVEL_PREFIX}
  COMMENT "Copying library files to python directory" )

endmacro(pybind_add_module)
