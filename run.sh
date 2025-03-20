#!/bin/bash

# Set script to stop on errors
set -e

# Compiler selection (optional, in case you need to override)
CXX=g++

# Set build directory
BUILD_DIR="build"

# Check if build directory exists, if not create it
if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

# Navigate to the build directory
cd "$BUILD_DIR"

# Run CMake (configure the project)
cmake ..

# Compile the project
make

# Run the compiled program
./RLLoadBalancer
