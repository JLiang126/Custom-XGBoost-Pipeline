cmake_minimum_required(VERSION 3.15)
project(XGBoostFromScratch)

# C++17 for modern features like unique_ptr and move semantics
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON) 
find_package(pybind11 REQUIRED)

set(SOURCES
    src/dataloader/DataMatrix.cpp
    src/engine/XGBoost.cpp
    src/objective/Loss.cpp
    src/XGTree/XGTree.cpp
    src/XGTree/XGNode.cpp
    python-packages/bindings.cpp
)

# Create the Python module.
pybind11_add_module(my_xgboost ${SOURCES})

target_include_directories(my_xgboost PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

# This connects the pybind11 logic to  binary
target_link_libraries(my_xgboost PRIVATE pybind11::module)