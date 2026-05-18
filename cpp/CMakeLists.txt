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
