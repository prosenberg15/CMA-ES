#!/usr/bin/env bash
# Simple build script for the CMA-ES project.
#   ./build.sh [source.cpp] [output_name]
# Defaults to building main.cpp. Examples:
#   ./build.sh                       # builds build/main
set -euo pipefail

# Eigen include path. Override with EIGEN_INC=...
EIGEN_INC="${EIGEN_INC:-/mnt/home/prosenberg/lib/eigen-5.0.0}"
CXX="${CXX:-clang++}"
CXXFLAGS="-std=c++23 -O2 -Wall -Wextra -I${EIGEN_INC} -I."

SRC="${1:-main.cpp}"
NAME="${2:-$(basename "${SRC%.*}")}"
OUT="build/${NAME}"

mkdir -p build
echo "Compiling ${SRC} -> ${OUT}"
"${CXX}" ${CXXFLAGS} "${SRC}" -o "${OUT}"
echo "Built ${OUT}"
