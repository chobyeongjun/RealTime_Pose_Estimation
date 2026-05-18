#!/usr/bin/env bash
# Build cpp/ extensions (Jetson).
#
# Output:
#   build/hwalker_shm_v2_writer*.so   → installed to src/perception/realtime/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo "=========================================="
echo "Build C++ extensions"
echo "  Repo: $REPO_ROOT"
echo "=========================================="

# Check pybind11 (pip install pybind11)
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo -e "${YELLOW}pybind11 not installed. Installing...${NC}"
    pip install --user pybind11
fi

# Check cmake
if ! command -v cmake >/dev/null 2>&1; then
    echo -e "${RED}cmake not found. Install: sudo apt-get install cmake${NC}"
    exit 1
fi

BUILD_DIR="$REPO_ROOT/build/cpp"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "── cmake configure ──"
cmake "$REPO_ROOT/cpp" -DCMAKE_BUILD_TYPE=Release

echo ""
echo "── cmake build ──"
cmake --build . -j"$(nproc)"

# Install: copy .so to src/perception/realtime/ so Python can import directly
INSTALL_DIR="$REPO_ROOT/src/perception/realtime"

SHM_SO=$(find . -name "hwalker_shm_v2_writer*.so" | head -1)
if [ -z "$SHM_SO" ]; then
    echo -e "${RED}Build failed: hwalker_shm_v2_writer.so not produced${NC}"
    exit 1
fi
cp -v "$SHM_SO" "$INSTALL_DIR/"

# Optional TRT runner (built only if TensorRT detected)
TRT_SO=$(find . -name "hwalker_trt_runner*.so" 2>/dev/null | head -1)
if [ -n "$TRT_SO" ]; then
    cp -v "$TRT_SO" "$INSTALL_DIR/"
    echo -e "${GREEN}  ✓ TRT runner built and installed${NC}"
else
    echo -e "${YELLOW}  ⚠ TRT runner NOT built (TensorRT/CUDA not detected by cmake)${NC}"
    echo "    See cmake output above. Continuing with shm_v2_writer only."
fi

# Optional CUDA preprocess kernel (built only if CUDA detected) — Sprint 1 Phase 2 Week 3
CUDA_PRE_SO=$(find . -name "hwalker_cuda_preprocess*.so" 2>/dev/null | head -1)
if [ -n "$CUDA_PRE_SO" ]; then
    cp -v "$CUDA_PRE_SO" "$INSTALL_DIR/"
    echo -e "${GREEN}  ✓ CUDA preprocess kernel built and installed${NC}"
else
    echo -e "${YELLOW}  ⚠ CUDA preprocess kernel NOT built (CUDA Toolkit not detected)${NC}"
fi

echo ""
echo -e "${GREEN}=== Build success ===${NC}"
echo "  shm_v2_writer:    $INSTALL_DIR/$(basename $SHM_SO)"
[ -n "$TRT_SO" ] && echo "  trt_runner:       $INSTALL_DIR/$(basename $TRT_SO)"
[ -n "$CUDA_PRE_SO" ] && echo "  cuda_preprocess:  $INSTALL_DIR/$(basename $CUDA_PRE_SO)"
echo ""
echo "Test import:"
echo "  cd $REPO_ROOT"
echo "  python3 -c 'from src.perception.realtime import hwalker_shm_v2_writer as w; print(w.Writer)'"
echo ""
echo "Run regression + benchmark:"
echo "  PYTHONPATH=src python3 tests/test_shm_v2_writer_cpp.py"
