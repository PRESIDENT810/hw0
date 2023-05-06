cmake_minimum_required(VERSION 3.20)
project(hw0 VERSION 1.00)

set(CMAKE_CXX_STANDARD 17)

#include(FetchContent)
#FetchContent_Declare(
#        googletest
#        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
#)
#FetchContent_MakeAvailable(googletest)

#enable_testing()
#
#include(GoogleTest)
#
#add_executable(simple_ml_ext_test src/simple_ml.h src/simple_ml_ext_test.cpp)
#gtest_discover_tests(simple_ml_ext_test)
#target_link_libraries(simple_ml_ext_test GTest::gtest_main)

add_library(simple_ml_ext SHARED src/simple_ml.h src/simple_ml_ext.cpp)
set_target_properties(simple_ml_ext PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(simple_ml_ext PROPERTIES OUTPUT_NAME simple_ml_ext)
set_target_properties(simple_ml_ext PROPERTIES PREFIX "")
set_target_properties(simple_ml_ext PROPERTIES SUFFIX ".so")

target_link_options(simple_ml_ext PUBLIC -undefined dynamic_lookup)

target_include_directories(simple_ml_ext PUBLIC /usr/local/lib/python3.11/site-packages/pybind11/include)
target_include_directories(simple_ml_ext PUBLIC /usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/include/python3.11)