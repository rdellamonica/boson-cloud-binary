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
    # Each test runs in its own scratch directory, since the solver writes
    # Simulation_* folders into the working directory.
    function(add_solver_test name)
        cmake_parse_arguments(ARG "" "" "LABELS;ARGS" ${ARGN})
        add_executable(${name} tests/${name}.cpp)
        target_link_libraries(${name} PRIVATE GravitationalMoleculeLib)
        target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
        set(work "${CMAKE_CURRENT_BINARY_DIR}/test-output/${name}")
        file(MAKE_DIRECTORY "${work}")
        add_test(NAME ${name} COMMAND ${name} ${ARG_ARGS})
        set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY "${work}" LABELS "${ARG_LABELS}")
    endfunction()

    # Unit tests of the building blocks
    add_solver_test(config_and_io LABELS unit)
    add_solver_test(rolling_average LABELS unit)

    # Regression tests against independent reference implementations
    add_solver_test(parallel_correctness LABELS regression)
    add_solver_test(spherical_absorption LABELS regression)
    add_solver_test(swave_projector LABELS regression)
    add_solver_test(isolated_bh LABELS regression)
    add_solver_test(orbital_ramp LABELS regression)
    add_solver_test(fixed_orbit_absorption LABELS regression)
    add_solver_test(flux_time_alignment LABELS regression)

    # Physical behaviour and conservation laws
    add_solver_test(physics LABELS physics)

    # Restart and command-line drivers
    add_solver_test(restart LABELS integration)
    add_solver_test(command_line LABELS integration
        ARGS $<TARGET_FILE:GravitationalMolecule> $<TARGET_FILE:IsolatedBH>)

    # Python: initial-state scripts and dashboard (skipped without the modules)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        add_test(NAME python_initial_states
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python/test_initial_states.py
                    --solver $<TARGET_FILE:GravitationalMolecule>)
        add_test(NAME python_dashboard
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python/test_dashboard.py
                    --solver $<TARGET_FILE:GravitationalMolecule>)
        set_tests_properties(python_initial_states python_dashboard PROPERTIES
            LABELS python SKIP_RETURN_CODE 77)
    endif()

    # Scaling benchmark (not a test): python3 tests/benchmark_parallel.py build/mode_benchmark
    add_executable(mode_benchmark tests/mode_benchmark.cpp)
    target_link_libraries(mode_benchmark PRIVATE GravitationalMoleculeLib)
endif()
