cmake_minimum_required(VERSION 3.22)
project(cvinfer VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# warning as error
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Compiler is ${CMAKE_CXX_COMPILER_ID}")
    add_compile_options(-Werror)
endif()

# get git branch and commit id
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Git branch: ${GIT_BRANCH}")
message(STATUS "Git commit id: ${GIT_COMMIT_ID}")

set(LIB_CVINFER cvinfer)
set(EXE_DEMO "demo")

add_library(${LIB_CVINFER} SHARED)
add_executable(${EXE_DEMO})

add_subdirectory(src)
add_subdirectory(test)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
set(TensorRT_DIR /opt/TensorRT-8.6.1.6)
# find_package(TensorRT REQUIRED)

# ffmpeg not support find_package
set(FFmpeg_INC /opt/ffmpeg/linux-install/include)
set(FFmpeg_LIB /opt/ffmpeg/linux-install/lib)
find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h ${FFmpeg_INC})
find_library(AVCODEC_LIBRARY avcodec ${FFmpeg_LIB})
find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h ${FFmpeg_INC})
find_library(AVFORMAT_LIBRARY avformat ${FFmpeg_LIB})
find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h ${FFmpeg_INC})
find_library(AVUTIL_LIBRARY avutil ${FFmpeg_LIB})
find_path(AVDEVICE_INCLUDE_DIR libavdevice/avdevice.h ${FFmpeg_INC})
find_library(AVDEVICE_LIBRARY avdevice ${FFmpeg_LIB})
target_include_directories(${LIB_CVINFER} PRIVATE ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR})
target_link_libraries(${LIB_CVINFER} PRIVATE ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${AVDEVICE_LIBRARY})

target_link_libraries(${LIB_CVINFER} PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${CUDNN_LIBRARIES} ${FFmpeg_LIBRARIES})
target_link_libraries(${EXE_DEMO} PRIVATE ${LIB_CVINFER})