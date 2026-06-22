#!/usr/bin/env bash
# Simple build script for the CMA-ES project.
#   ./build.sh [source.cpp] [output_name]
# Defaults to building main.cpp. Examples:
#   ./build.sh                       # builds build/main
#   ./build.sh test_objective.cpp    # builds build/test_objective
set -euo pipefail

# Eigen include path (Homebrew on Apple Silicon). Override with EIGEN_INC=...
EIGEN_INC="${EIGEN_INC:-/opt/homebrew/include/eigen3}"
CXX="${CXX:-clang++}"
CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -I${EIGEN_INC} -I."

SRC="${1:-main.cpp}"
NAME="${2:-$(basename "${SRC%.*}")}"
OUT="build/${NAME}"

mkdir -p build
echo "Compiling ${SRC} -> ${OUT}"
"${CXX}" ${CXXFLAGS} "${SRC}" -o "${OUT}"
echo "Built ${OUT}"
