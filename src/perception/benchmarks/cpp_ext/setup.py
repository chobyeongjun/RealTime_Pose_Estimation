"""
C++ 후처리 확장 모듈 빌드 스크립트
===================================
빌드:
    cd benchmarks/cpp_ext
    pip install pybind11
    python setup.py build_ext --inplace

또는 개발 모드:
    pip install -e .
"""

import os
import sys
from setuptools import setup, Extension

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    print("pybind11이 필요합니다: pip install pybind11")
    sys.exit(1)

ext_modules = [
    Extension(
        "pose_postprocess_cpp",
        sources=["pose_postprocess.cpp"],
        include_dirs=[pybind11_include],
        language="c++",
        extra_compile_args=[
            "-std=c++17",
            "-O3",                    # 최대 최적화
            "-ffast-math",            # 빠른 수학 연산
            "-fno-finite-math-only",  # NaN/Inf 체크 복구 (depth 무효 판별에 필수)
            "-march=native",          # 현재 CPU 아키텍처 최적화 (Jetson ARM NEON 활용)
            "-fPIC",
        ],
    ),
]

setup(
    name="pose_postprocess_cpp",
    version="1.0.0",
    description="Pose estimation 후처리 C++ 확장 모듈",
    ext_modules=ext_modules,
)
