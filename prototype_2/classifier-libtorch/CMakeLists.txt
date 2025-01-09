cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(federated-classifier)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(federated-classifier main.cpp classifier.cpp)
target_link_libraries(federated-classifier "${TORCH_LIBRARIES}")
set_property(TARGET federated-classifier PROPERTY CXX_STANDARD 17)
