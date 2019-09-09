cmake_minimum_required(VERSION 3.0)
project(imgproc-tools)

option(BUILD_PYTHON_BINDING "Build Python binding." ON)
option(BUILD_SHARED "Build shared library." OFF)

find_package(PkgConfig)
pkg_search_module(OpenCV opencv)
if(NOT OpenCV_FOUND)
    pkg_search_module(OpenCV REQUIRED opencv4)
    set(OpenCV_INCLUDE_DIRS ${OpenCV4_INCLUDE_DIRS})
endif()

set(OpenCV_LIBRARIES opencv_core)

file(GLOB SOURCES_LIST ${PROJECT_SOURCE_DIR}/src/cpp/*.cpp)

if(BUILD_SHARED)
    add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST})
else()
    add_library(${PROJECT_NAME} STATIC ${SOURCES_LIST})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS}
                                           PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
                                                 CXX_STANDARD 14
                                                 CXX_STANDARD_REQUIRED TRUE
                                                 LINKER_LANGUAGE "CXX"
                                                 POSITION_INDEPENDENT_CODE ON)

if(BUILD_PYTHON_BINDING)
    string(REPLACE "-" "_" PYTHON_MODULE_NAME ${PROJECT_NAME})

    add_definitions(-DMODULE_NAME=${PYTHON_MODULE_NAME})

    list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules")

    find_package(PythonLibs REQUIRED)
    find_package(NumPy REQUIRED)
    find_package(pybind11 REQUIRED)

    file(GLOB SOURCES_LIST ${PROJECT_SOURCE_DIR}/src/python/*.cpp)

    python_add_module(${PYTHON_MODULE_NAME} ${SOURCES_LIST})

    target_include_directories(${PYTHON_MODULE_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS}
                                                     PRIVATE ${PYTHON_NUMPY_INCLUDE_DIR}
                                                     PRIVATE ${PROJECT_NAME})

    target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE ${PROJECT_NAME} ${PYTHON_LIBRARIES})
endif()
