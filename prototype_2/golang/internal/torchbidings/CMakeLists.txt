cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(classifier)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(classifier SHARED classifier.cpp)
target_include_directories(classifier PRIVATE json)
target_link_libraries(classifier "${TORCH_LIBRARIES}")
set_property(TARGET classifier PROPERTY CXX_STANDARD 17)
