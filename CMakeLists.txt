cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_executable(MyProject project.cu)
target_link_libraries(MyProject "${TORCH_LIBRARIES}")
set_property(TARGET MyProject PROPERTY CXX_STANDARD 14)

# If CUDA is needed:
set(CMAKE_CUDA_STANDARD 14)
enable_language(CUDA)
