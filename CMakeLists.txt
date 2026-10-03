cmake_minimum_required(VERSION 3.16)
project(numeric_flow VERSION 0.1.0 LANGUAGES CXX)

add_library(numeric_flow cpp/src/solver.cpp)
target_include_directories(numeric_flow PUBLIC cpp/include)
target_compile_features(numeric_flow PUBLIC cxx_std_17)
set_target_properties(numeric_flow PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(numeric_flow PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(numeric-flow cpp/src/main.cpp)
target_link_libraries(numeric-flow PRIVATE numeric_flow)

include(CTest)
if(BUILD_TESTING)
  add_executable(numeric-flow-tests cpp/tests/test_solver.cpp)
  target_link_libraries(numeric-flow-tests PRIVATE numeric_flow)
  add_test(NAME cpp-numerics COMMAND numeric-flow-tests)

  find_package(Python3 3.10 COMPONENTS Interpreter)
  if(Python3_FOUND)
    add_test(NAME python-and-parity
      COMMAND ${CMAKE_COMMAND} -E env NUMERIC_FLOW_CPP=$<TARGET_FILE:numeric-flow>
              ${Python3_EXECUTABLE} -m unittest discover -s tests -v)
    set_tests_properties(python-and-parity PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
endif()
