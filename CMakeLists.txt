cmake_minimum_required(VERSION 3.16)
project(GravitationalMolecule LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)

# ----------------------------------------------------------------------
# OpenMP
# ----------------------------------------------------------------------

# AppleClang needs an external OpenMP runtime. Search installed locations
# without invoking Homebrew (which can update caches or require network access).
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(LIBOMP_ROOT "" CACHE PATH "Prefix containing libomp/include and libomp/lib")
    find_path(LIBOMP_INCLUDE_DIR omp.h
        HINTS "${LIBOMP_ROOT}/include" /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
    find_library(LIBOMP_LIBRARY omp
        HINTS "${LIBOMP_ROOT}/lib" /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    if(NOT LIBOMP_INCLUDE_DIR OR NOT LIBOMP_LIBRARY)
        message(FATAL_ERROR "OpenMP runtime not found. Install libomp and/or set LIBOMP_ROOT.")
    endif()
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_LIBRARY}")
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Core library
add_library(GravitationalMoleculeLib
    src/CrankNicolsonModes.cpp
    src/CrankNicolsonModes_absorption.cpp
    src/CrankNicolsonModes_evolution.cpp
    src/CrankNicolsonModes_extraction.cpp
    src/CrankNicolsonModes_hamiltonian.cpp
    src/CrankNicolsonModes_matrices.cpp
    src/CrankNicolsonModes_outgoing.cpp
    src/CrankNicolsonModes_restart.cpp
    src/CrankNicolsonModes_utils.cpp
    src/RollingPhase.cpp
    src/InitialData.cpp
    src/DisplacedAtom.cpp
    src/Definitions.cpp
    src/Utils.cpp
)

target_include_directories(GravitationalMoleculeLib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_SOURCE_DIR}/external
)

if(LIBOMP_INCLUDE_DIR)
    target_include_directories(GravitationalMoleculeLib SYSTEM PUBLIC "${LIBOMP_INCLUDE_DIR}")
endif()

target_link_libraries(GravitationalMoleculeLib
    PUBLIC
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
        GSL::gsl
        GSL::gslcblas
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Executables
add_executable(GravitationalMolecule src/main.cpp)
target_link_libraries(GravitationalMolecule PRIVATE GravitationalMoleculeLib)

add_executable(IsolatedBH src/isolated_bh_main.cpp)
target_link_libraries(IsolatedBH PRIVATE GravitationalMoleculeLib)

# Install rules
install(TARGETS GravitationalMolecule IsolatedBH GravitationalMoleculeLib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

# Optional API documentation: cmake --build build --target docs
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/docs/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation in docs/html"
        VERBATIM)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(isolated_bh tests/isolated_bh.cpp)
    target_link_libraries(isolated_bh PRIVATE GravitationalMoleculeLib)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/isolated-test-output")
    add_test(NAME isolated_bh COMMAND isolated_bh)
    set_tests_properties(isolated_bh PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/isolated-test-output")
    add_executable(parallel_correctness tests/parallel_correctness.cpp)
    target_link_libraries(parallel_correctness PRIVATE GravitationalMoleculeLib)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-output")
    add_test(NAME parallel_correctness COMMAND parallel_correctness)
    set_tests_properties(parallel_correctness PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-output")
    add_executable(orbital_ramp tests/orbital_ramp.cpp)
    target_link_libraries(orbital_ramp PRIVATE GravitationalMoleculeLib)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/orbital-test-output")
    add_test(NAME orbital_ramp COMMAND orbital_ramp)
    set_tests_properties(orbital_ramp PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/orbital-test-output")
    add_executable(fixed_orbit_absorption tests/fixed_orbit_absorption.cpp)
    target_link_libraries(fixed_orbit_absorption PRIVATE GravitationalMoleculeLib)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/absorption-test-output")
    add_test(NAME fixed_orbit_absorption COMMAND fixed_orbit_absorption)
    set_tests_properties(fixed_orbit_absorption PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/absorption-test-output")
    add_executable(spherical_absorption tests/spherical_absorption.cpp)
    target_link_libraries(spherical_absorption PRIVATE GravitationalMoleculeLib)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/spherical-test-output")
    add_test(NAME spherical_absorption COMMAND spherical_absorption)
    set_tests_properties(spherical_absorption PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/spherical-test-output")
    add_executable(swave_projector tests/swave_projector.cpp)
    target_link_libraries(swave_projector PRIVATE GravitationalMoleculeLib)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/swave-test-output")
    add_test(NAME swave_projector COMMAND swave_projector)
    set_tests_properties(swave_projector PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/swave-test-output")
    add_executable(mode_benchmark tests/mode_benchmark.cpp)
    target_link_libraries(mode_benchmark PRIVATE GravitationalMoleculeLib)
    add_executable(flux_time_alignment tests/flux_time_alignment.cpp)
    target_link_libraries(flux_time_alignment PRIVATE GravitationalMoleculeLib)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/flux-test-output")
    add_test(NAME flux_time_alignment COMMAND flux_time_alignment)
    set_tests_properties(flux_time_alignment PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/flux-test-output")
endif()
