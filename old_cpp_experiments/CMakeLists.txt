cmake_minimum_required(VERSION 3.24)
project(crunch)

set(CMAKE_CXX_STANDARD 20)

include_directories(/opt/homebrew/Cellar/matplotplusplus/1.1.0_3/include .)

add_executable(crunch
        main.cpp
        wallace.cpp
        wallace.h
        utilities.cpp
        utilities.h
        Case.cpp
        Case.h
        Plotter.cpp
        Plotter.h
        math/activation_functions.cpp
        math/activation_functions.h
        math/metrics.cpp
        math/metrics.h
        network/Training.cpp
        network/Training.h
        network/Neuron.cpp
        network/Neuron.h
        network/Network.cpp
        network/Network.h
        network/Weights.cpp
        network/Weights.h
)

target_compile_options(crunch PUBLIC ${CMAKE_C_FLAGS} -O3)

target_link_libraries(crunch /opt/homebrew/Cellar/matplotplusplus/1.1.0_3/lib/libmatplot.dylib)
