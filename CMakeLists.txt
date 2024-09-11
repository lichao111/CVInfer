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


set(GIT_BRANCH_NAME ${GIT_BRANCH})
set(GIT_COMMIT_ID ${GIT_COMMIT_ID})

configure_file(${CMAKE_SOURCE_DIR}/src/tools/version.h.in  ${CMAKE_SOURCE_DIR}/src/tools/version.h @ONLY)

set(LIB_CVINFER cvinfer)
set(EXE_DEMO demo)
set(LIB_CUDA_PLUGIN plugin_list)

add_library(${LIB_CVINFER} SHARED)
add_executable(${EXE_DEMO})

add_subdirectory(src)
enable_testing()
add_subdirectory(test)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
set(TensorRT_DIR /opt/TensorRT-8.6.1.6)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS ${TensorRT_DIR})
find_library(TENSORRT_LIBRARY nvinfer PATHS ${TensorRT_DIR})
find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser PATHS ${TensorRT_DIR})
find_library(TENSORRT_LIBRARY_PLUGIN nvinfer_plugin PATHS ${TensorRT_DIR})

# Check if TensorRT was found
if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_LIBRARY_PLUGIN)
    message(FATAL_ERROR "TensorRT libraries or include directories not found.")
endif()

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
find_path(AVSWSCALE_INCLUDE_DIR libswscale/swscale.h ${FFmpeg_INC})
find_library(AVSWSCALE_LIBRARY swscale ${FFmpeg_LIB})

target_include_directories(${LIB_CVINFER} PUBLIC 
    ${AVCODEC_INCLUDE_DIR} 
    ${AVFORMAT_INCLUDE_DIR} 
    ${AVUTIL_INCLUDE_DIR} 
    ${AVDEVICE_INCLUDE_DIR} 
    ${AVSWSCALE_INCLUDE_DIR} 
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR})

target_link_libraries(${LIB_CVINFER} PUBLIC 
    ${AVCODEC_LIBRARY} 
    ${AVFORMAT_LIBRARY} 
    ${AVUTIL_LIBRARY} 
    ${AVDEVICE_LIBRARY} 
    ${AVSWSCALE_LIBRARY} 
    ${LIB_CUDA_PLUGIN})

target_link_libraries(${LIB_CVINFER} PUBLIC 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES} 
    ${TensorRT_LIBRARIES} 
    ${CUDNN_LIBRARIES} 
    ${FFmpeg_LIBRARIES}
    ${TENSORRT_LIBRARY} 
    ${TENSORRT_LIBRARY_PLUGIN}
    ${TENSORRT_ONNX_PARSER_LIBRARY})

target_link_libraries(${EXE_DEMO} PRIVATE ${LIB_CVINFER})

add_custom_target(personball_mini
    COMMAND ./${EXE_DEMO} personball_mini
    DEPENDS ${EXE_DEMO}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(personball
    COMMAND ./${EXE_DEMO} personball
    DEPENDS ${EXE_DEMO}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(yolov7
    COMMAND ./${EXE_DEMO} yolov7
    DEPENDS ${EXE_DEMO}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(yolov5s
    COMMAND ./${EXE_DEMO} yolov5s
    DEPENDS ${EXE_DEMO}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)