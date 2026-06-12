cmake_minimum_required(VERSION 3.16)

project(deepcircus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)

add_executable(main
    gen.cpp
    solution.cpp
    main.cpp
)

target_compile_definitions(main PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_include_directories(main PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(main PRIVATE Threads::Threads)
