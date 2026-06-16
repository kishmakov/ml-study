cmake_minimum_required(VERSION 3.16)

project(deepcircus_generator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(generator
    decision_tree.cpp
    decision_tree.h
    generator.cpp
    generator.h
    small_bitness.cpp
    small_bitness.h
)

target_include_directories(generator PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
