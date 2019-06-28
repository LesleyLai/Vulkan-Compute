cmake_minimum_required(VERSION 3.8)

project(VulkanCompute VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include("compiler")
include("clangformat")

if (BP_ENABLE_CLANG_TIDY)
include("ClangTidy")
endif()

if (BP_ENABLE_CPPCHECK)
include("cppcheck")
endif()

if(MSVC)
    add_compile_definitions(NOMINMAX)
endif()


# Conan package manager
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
                  "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)


conan_check(VERSION 1.4.0 REQUIRED)
conan_add_remote(NAME bincrafters INDEX 1
URL https://api.bintray.com/conan/bincrafters/public-conan)

conan_cmake_run(CONANFILE conanfile.txt
                BASIC_SETUP CMAKE_TARGETS
                BUILD missing)

add_subdirectory(src)
