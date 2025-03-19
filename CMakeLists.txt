cmake_minimum_required(VERSION 3.25.1)
project(RLLoadBalancer)

set(CMAKE_CXX_STANDARD 20)

add_executable(RLLoadBalancer main.cpp
        Environment/Environment.h
        Environment/Task.h
        Environment/Task.cpp
        Environment/Environment.cpp
        Environment/Processor.cpp
        Environment/Processor.h
)
