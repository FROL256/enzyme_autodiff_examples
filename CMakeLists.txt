cmake_minimum_required (VERSION 3.15)
project (enzyme_test)

set(CLANG_VERSION "16" CACHE STRING "clang and LLVM version to use")
set(CMAKE_C_COMPILER   "/usr/bin/clang-${CLANG_VERSION}")   # $ export CC=/usr/bin/clang-16
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-${CLANG_VERSION}") # $ export CXX=/usr/bin/clang++-16
set(ENZYME_PLUGIN_DLL  "/home/frol/PROG/enzyme_main/enzyme/build/Enzyme/ClangEnzyme-${CLANG_VERSION}.so")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DUSE_ENZYME)

add_executable(test main.cpp)
target_compile_options(test PUBLIC "-fplugin=${ENZYME_PLUGIN_DLL}")