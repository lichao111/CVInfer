# for ut
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
set(EXE_UT "runTests")
add_executable(${EXE_UT})
file(GLOB EXE_UT_SRC "test_*.cpp")
message(STATUS "EXE_UT_SRC: ${EXE_UT_SRC}")
target_sources(${EXE_UT} PRIVATE ${EXE_UT_SRC})
target_include_directories(runTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(runTests ${LIB_CVINFER} ${GTEST_LIBRARIES} pthread)
add_test(NAME ctest COMMAND ${EXE_UT} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)
# for demo
file(GLOB DEMO_SRC "demo.cpp")
target_sources(${EXE_DEMO} PRIVATE ${DEMO_SRC})

