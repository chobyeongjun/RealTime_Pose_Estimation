# cpp/CMakeLists.txt — H-Walker C++ extensions (Sprint 1)
#
# Build (Jetson):
#   cd ~/realtime-vision-control
#   bash scripts/build_cpp.sh
#
# Output:
#   build/hwalker_shm_v2_writer.cpython-310-aarch64-linux-gnu.so
#   → installed to src/perception/realtime/ as Python import target
cmake_minimum_required(VERSION 3.16)
project(hwalker_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Aggressive optimization for hot path
add_compile_options(-Wall -Wextra -O3 -fno-omit-frame-pointer)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*")
    # Jetson Orin is Cortex-A78AE (ARMv8.2-A)
    add_compile_options(-march=armv8.2-a+crypto+fp16)
endif()

# ─── pybind11 ─────────────────────────────────────────────────────────────
# Prefer system pip install (`pip install pybind11`) which provides cmake config.
# Fallback: find via Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_status
)
if(NOT _pybind11_status EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. Install: pip install pybind11")
endif()
find_package(pybind11 REQUIRED CONFIG)

# ─── SHM v2 writer module ─────────────────────────────────────────────────
pybind11_add_module(hwalker_shm_v2_writer
    src/shm_v2_writer.cpp
    src/python_bindings.cpp
)
target_include_directories(hwalker_shm_v2_writer
    PRIVATE include
)
target_link_libraries(hwalker_shm_v2_writer
    PRIVATE rt   # POSIX shm_open requires librt on some systems
)

# ─── TRT inference runner — Sprint 1 Phase 2 Week 1 ───────────────────────
# Detect TensorRT 10.x (Jetson installs to /usr/lib/aarch64-linux-gnu)
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS /usr/include
          /usr/include/aarch64-linux-gnu
          /usr/include/x86_64-linux-gnu
          /usr/local/tensorrt/include
          /opt/nvidia/tensorrt/include
)
find_library(TENSORRT_INFER_LIB
    NAMES nvinfer
    HINTS /usr/lib/aarch64-linux-gnu
          /usr/lib/x86_64-linux-gnu
          /usr/local/tensorrt/lib
          /opt/nvidia/tensorrt/lib
)

# CUDA toolkit (cudart for cudaStreamSynchronize)
find_package(CUDAToolkit QUIET)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_INFER_LIB AND CUDAToolkit_FOUND)
    message(STATUS "TensorRT found: ${TENSORRT_INFER_LIB}")
    message(STATUS "TensorRT headers: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")

    pybind11_add_module(hwalker_trt_runner
        src/trt_runner.cpp
        src/trt_binding.cpp
    )
    target_include_directories(hwalker_trt_runner
        PRIVATE include
                ${TENSORRT_INCLUDE_DIR}
                ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(hwalker_trt_runner
        PRIVATE ${TENSORRT_INFER_LIB}
                CUDA::cudart
    )
else()
    message(WARNING "TensorRT or CUDA not found — skipping hwalker_trt_runner build.")
    if(NOT TENSORRT_INCLUDE_DIR)
        message(WARNING "  NvInfer.h not found. Install: sudo apt install libnvinfer-headers-dev")
    endif()
    if(NOT TENSORRT_INFER_LIB)
        message(WARNING "  libnvinfer.so not found. Install: sudo apt install libnvinfer-dev")
    endif()
    if(NOT CUDAToolkit_FOUND)
        message(WARNING "  CUDA Toolkit not found.")
    endif()
endif()
